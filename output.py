#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, importlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt

# === 路径拼上你的工程（与之前一致） ===
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

# === 你已有的 MoE 与公共模块 ===
# TODO: 把下面这一行改成你工程里 DiME 的真实路径
from src.dime.dime_model import DiME

from transformers import AutoTokenizer, BertModel, CLIPModel

def _to_logits(x):
    # 兼容返回 logits 或 (logits, gate_loss)
    if isinstance(x, (tuple, list)):
        return x[0]
    return x

def infer_fusion_hparams(moe_sd, num_modalities=2):
    # d_model, seq_len from pos_embed
    pos_key = "interaction_experts.0.fusion_model.pos_embed"
    assert pos_key in moe_sd, f"missing key: {pos_key}"
    _, L, H = moe_sd[pos_key].shape

    # num_layers: count of network.k.norm1.weight
    prefix = "interaction_experts.0.fusion_model.network."
    layer_ids = sorted({
        int(k.split(".")[4])
        for k in moe_sd.keys()
        if k.startswith(prefix) and ".norm1.weight" in k
    })
    num_layers = len(layer_ids) if layer_ids else 1

    # num_classes 先未知，这一步返回 None（后面别名探测后再定）
    num_heads = 8  # 不影响权重加载
    return dict(d_model=H, seq_len=L, num_layers=num_layers, num_heads=num_heads,
                num_classes=None, num_modalities=num_modalities)


# ====== 你贴出来的这些基础模块（保持命名一致，方便 state_dict 对齐） ======
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super().__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

class PatchEmbeddings(nn.Module):
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        import math
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches * patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)
    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        return self.projection(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"embed dim {dim} must be divisible by num_heads {num_heads}"
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,   # 输入输出: [B, N, C]
            bias=qkv_bias,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv, attn_mask=None):
        # x, kv: [B, N, C]
        out, _ = self.mha(x, kv, kv, attn_mask=attn_mask, need_weights=False)
        return self.proj_drop(out)  # [B, N, C]

# 维持与您代码一致的 TransformerEncoderLayer（去掉稀疏MoE依赖，兼容加载）
class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_experts, num_routers, d_model, num_head, dropout=0.1,
                 activation=nn.GELU, hidden_times=2, mlp_sparse=False, self_attn=True, top_k=2, gate="GShardGate", **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn
        # 简化：这里用密集 MLP；若你的 ckpt 用了稀疏MLP，state_dict 里也会对齐到 self.mlp.network.* 权重（同名）
        self.mlp = MLP(input_dim=d_model, hidden_dim=d_model*hidden_times, output_dim=d_model, num_layers=2, activation=nn.GELU(), dropout=dropout)
    def forward(self, x_list):
        # x_list: List[Tensor[B, P_i, H]]
        chunk = [item.shape[1] for item in x_list]
        x_cat = torch.cat(x_list, dim=1)               # [B, sumP, H]
        x = self.norm1(x_cat)
        x = self.attn(x, x) + self.dropout1(x)         # self-attn
        x_split = torch.split(x, chunk, dim=1)
        out_list = []
        for i, xi in enumerate(x_split):
            out = xi + self.dropout2(self.mlp(self.norm2(xi)))
            out_list.append(out)
        return out_list

# === 融合模型骨架（字段名与 ckpt 保持：pos_embed / network / head / gate_loss） ===
class FusionTransformer(nn.Module):
    def __init__(self, num_modalities, d_model, num_layers, num_heads, seq_len, num_classes, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.d_model = d_model
        self.seq_len = seq_len
        # 位置嵌入按 ckpt 形状注册
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # 堆叠编码层
        self.network = nn.ModuleList([
            TransformerEncoderLayer(
                num_experts=0, num_routers=0, d_model=d_model, num_head=num_heads,
                dropout=dropout, activation=nn.GELU, hidden_times=2, mlp_sparse=False, self_attn=True
            ) for _ in range(num_layers)
        ])
        # 池化 + 线性分类头（保持名字 head）
        self.head = nn.Linear(d_model * num_modalities, num_classes)

    def forward(self, inputs):
        # inputs: List[Tensor[B, P_i, H]]，长度=num_modalities
        B = inputs[0].size(0)
        # 给每个模态分配一段位置编码切片
        lens = [x.shape[1] for x in inputs]
        assert sum(lens) <= self.seq_len, f"sum patches {sum(lens)} > pos_embed len {self.seq_len}"
        pos_slices = torch.split(self.pos_embed[:, :sum(lens), :], lens, dim=1)
        x_list = [inp + pos for inp, pos in zip(inputs, pos_slices)]
        for layer in self.network:
            x_list = layer(x_list)  # 每层保持列表接口
        # 各模态做 mean pooling，再拼接
        pooled = [x.mean(dim=1) for x in x_list]     # List[B, H]
        feat = torch.cat(pooled, dim=1)              # [B, H * M]
        logits = self.head(feat)                     # [B, C]
        return logits

    def gate_loss(self):
        # 兼容 MoE 接口：如果没有稀疏门控，就返回 0
        return torch.tensor(0.0, device=self.pos_embed.device)

# ====== 编码器：与你的预处理一致 ======
class BertEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers_enc=1, patch=True, num_patches=16):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        mlp = MLP(768, hidden_dim, hidden_dim, num_layers_enc)
        if patch:
            self.proj = nn.Sequential(mlp, PatchEmbeddings(feature_size=hidden_dim, num_patches=num_patches, embed_dim=hidden_dim))
        else:
            self.proj = mlp
    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
        return self.proj(pooled)

class CLIPImageEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers_enc=1, patch=True, num_patches=16):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        vision_width = self.clip.config.vision_config.hidden_size
        mlp = MLP(vision_width, hidden_dim, hidden_dim, num_layers_enc)
        if patch:
            self.proj = nn.Sequential(mlp, PatchEmbeddings(feature_size=hidden_dim, num_patches=num_patches, embed_dim=hidden_dim))
        else:
            self.proj = mlp
    def forward(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.proj(pooled)

# ====== 预处理与可视化 ======
def load_image_tensor(path):
    tfs = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711)),
    ])
    im = Image.open(path)
    if (im.mode == "P" and "transparency" in im.info) or im.mode == "RGBA":
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,255,255,255)), im.convert("RGBA")).convert("RGB")
    else:
        im = im.convert("RGB")
    ten = tfs(im).unsqueeze(0)
    return ten, im

def tokenize_text(tweet, cot=""):
    tok = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    sep = tok.sep_token or "[SEP]"
    text = tweet.strip() if not cot else f"{tweet.strip()} {sep} {cot.strip()}"
    return tok(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

@torch.no_grad()
def score_from_logits(logits_or_tuple, target=None, use_softmax=False):
    logits = _to_logits(logits_or_tuple)
    if target is None:
        target = logits.argmax(dim=-1).item()
    s = torch.softmax(logits, dim=-1)[0, target] if use_softmax else logits[0, target]
    return s, int(target)


def make_baseline_like(x, mode="noise"):
    if mode == "zeros": return torch.zeros_like(x)
    if mode == "mean":  return x.mean().expand_as(x)
    return torch.randn_like(x)

def sliding_windows(H,W,win,stride):
    ys = list(range(0, max(H-win,0)+1, stride)) or [0]
    xs = list(range(0, max(W-win,0)+1, stride)) or [0]
    return [(i,j) for i in ys for j in xs]

def overlay_and_save(pil_img, heat_2hw, out_path, alpha=0.45, dpi=150):
    base = pil_img.resize((heat_2hw.shape[1], heat_2hw.shape[0]), Image.BILINEAR)
    base_np = np.asarray(base).astype(np.float32)/255.0
    hm = Image.fromarray((heat_2hw*255).astype(np.uint8)).resize((base_np.shape[1], base_np.shape[0]), Image.BILINEAR)
    hm = np.array(hm)/255.0
    plt.figure(figsize=(base_np.shape[1]/dpi*2, base_np.shape[0]/dpi*2), dpi=dpi)
    plt.imshow(base_np); plt.imshow(hm, cmap="jet", alpha=alpha); plt.axis('off'); plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0); plt.close()

# ====== 关键：每个专家的图像遮挡热力图 ======
def occlusion_heatmap_for_expert(moe_model, expert_index, inputs_embed_list, img_encoder, img_tensor,
                                 img_modality_index=1, target_class=None, window=32, stride=16,
                                 baseline_mode="noise", score_use_softmax=False):
    device = next(moe_model.parameters()).device
    expert = moe_model.interaction_experts[expert_index]
    base_logits = expert.forward(inputs_embed_list)
    base_score, tgt = score_from_logits(base_logits, target=target_class, use_softmax=score_use_softmax)

    # 每个滑窗
    logits = expert.forward_with_replacement(
        inputs_embed_list, replace_index=img_modality_index, replacement_tensor=img_embed_repl
    )
    score, _ = score_from_logits(logits, target=tgt, use_softmax=score_use_softmax)

    x = img_tensor.to(device)  # [1,3,H,W]
    _, _, H, W = x.shape
    heat = torch.zeros((H,W), device=device); counts = torch.zeros((H,W), device=device)

    for (i,j) in sliding_windows(H,W,window,stride):
        repl = x.clone()
        repl[:,:,i:i+window, j:j+window] = make_baseline_like(repl[:,:,i:i+window, j:j+window], baseline_mode)
        with torch.no_grad():
            img_embed_repl = img_encoder(repl)
        logits = expert.forward_with_replacement(inputs_embed_list, replace_index=img_modality_index, replacement_tensor=img_embed_repl)
        score, _ = score_from_logits(logits, target=tgt, use_softmax=score_use_softmax)
        delta = (base_score - score).clamp(min=0)
        heat[i:i+window, j:j+window] += delta; counts[i:i+window, j:j+window] += 1

    counts = torch.where(counts==0, torch.ones_like(counts), counts)
    heat = (heat / counts); heat = heat - heat.min()
    if heat.max() > 0: heat = heat / heat.max()
    return heat.detach().cpu().numpy(), tgt

def _find_classifier_for_expert(moe_sd, expert_idx, d_model, num_modalities=2,
                                max_c=1000, prefer_small_c=True):
    """
    在 interaction_experts.{e}.fusion_model.* 下找看起来像分类层的 2D 权重:
    返回 (weight_key, bias_key 或 None, C, D)
    规则：
      - 只看 .weight 且 tensor.ndim==2
      - 优先 out_features C 较小的（<= max_c），且 in_features D 最大的（更像最后一层）
    """
    base = f"interaction_experts.{expert_idx}.fusion_model."
    candidates = []
    for k, v in moe_sd.items():
        if not k.startswith(base): 
            continue
        if not k.endswith(".weight"): 
            continue
        if v.ndim != 2: 
            continue
        C, D = v.shape
        # 跳过明显是中间层但 C 特别大那种；C<=max_c 作为经验阈
        score_c = (C if prefer_small_c else 0)
        candidates.append((k, C, D, score_c))
    if not candidates:
        return None, None, None, None

    # 先按 C 升序（小类数优先），再按 D 降序（输入维度最大的）
    candidates.sort(key=lambda x: (x[1], -x[2]))
    w_key, C, D, _ = candidates[0]

    b_key = w_key[:-7] + ".bias"  # 替换 .weight -> .bias
    if b_key not in moe_sd:
        b_key = None
    return w_key, b_key, C, D


def alias_classifier_to_head(moe_sd, d_model, num_experts, num_modalities=2):
    """
    为每个 expert 找到分类层权重，并在 state_dict 中添加 head 别名键：
      interaction_experts.{e}.fusion_model.head.weight / .bias
    返回 (新的state_dict, num_classes)
    """
    new_sd = dict(moe_sd)  # 浅拷贝
    detected_C = None
    for e in range(num_experts):
        w_key, b_key, C, D = _find_classifier_for_expert(
            moe_sd, e, d_model, num_modalities=num_modalities
        )
        if w_key is None:
            print(f"[warn] expert {e}: 未找到可疑的分类层，跳过别名映射")
            continue
        # 写入别名
        alias_w = f"interaction_experts.{e}.fusion_model.head.weight"
        new_sd[alias_w] = moe_sd[w_key].clone()
        if b_key is not None:
            alias_b = f"interaction_experts.{e}.fusion_model.head.bias"
            new_sd[alias_b] = moe_sd[b_key].clone()
        print(f"[alias] expert {e}: {w_key}  ->  {alias_w}  (C={C}, D={D})")
        detected_C = detected_C or C
        # 也可以 sanity check：不同 expert 的 C 不一致时取最常见的一个
    if detected_C is None:
        raise RuntimeError("无法为任何 expert 找到分类层；请打印 keys 看看命名。")
    return new_sd, detected_C

# ====== 主流程 ======
def main():
    ap = argparse.ArgumentParser("Per-expert heatmaps with transformer fusion (auto-reconstruct)")
    ap.add_argument("--ckpt",type=str, default="./saves/dime/transformer/mmcsd/JB_CLIP_prompt_NoSyn_NoRdn_seed_1_modality_TS_train_epochs_80_val_f1_0.79.pth")
    ap.add_argument("--image", type=str,default="/home/zbw/project/SILO-MM/I2MoE-main/1311025564149403649_0.jpg")
    ap.add_argument("--tweet",type=str, default="""Joe Biden says he is super ready for tonight's Presidential Debate. debate2020 2020debate trumpbidendebate presidentialdebate presidentialdebate2020 """)
    ap.add_argument("--cot", default="""The provided sentence, "Joe Biden says he is super ready for tonight's Presidential Debate. #debate2020 #2020debate #trumpbidendebate #presidentialdebate #presidentialdebate2020," expresses a confident and positive attitude towards Joe Biden in relation to the topic of his preparedness for the Presidential Debate. The phrase "super ready" suggests a high level of confidence and readiness.

The image, however, presents a different attitude. It appears to be edited in a way to satirize or mock Joe Biden. The image shows what seems to be notes on his hands that read “It’s 2020,” “I’m running for President of the USA,” and “My name is Joe Biden, Look at other hand!” This is a comedic device often used to imply that someone might be forgetful or needs simple reminders for what would normally be considered common knowledge, especially in the context of a high-stakes event like a presidential debate. 

The editing of the image alongside the expression on the person's face creates a caricatured and humorous depiction, which contrasts with the earnestness of the text's expression of preparedness. The combined message of the text and image suggests a discrepancy between the public portrayal of confidence by Joe Biden or his team regarding the debate, and a satirical commentary that may question his readiness or abilities.

Remember, this analysis is purely based on the contents of the text and the image provided and is not a reflection of any actual events or the capabilities of Joe Biden.""")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--out_dir", default="./expert_heatmaps")
    ap.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--baseline", choices=["noise","zeros","mean"], default="mean")
    ap.add_argument("--use_softmax", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    moe_sd = ckpt["ensemble_model"]
    enc_pack = ckpt["encoder_dict"]

    # 1) 推断融合超参并构建 fusion_model
    hp = infer_fusion_hparams(moe_sd, num_modalities=2)

    # 2) 别名映射分类层，并拿到 num_classes
    #    先估一个专家数（从 keys 推断 interaction_experts.* 的最大索引）
    expert_ids = sorted({
        int(k.split(".")[1]) 
        for k in moe_sd.keys() 
        if k.startswith("interaction_experts.")
    })
    num_experts = (max(expert_ids) + 1) if expert_ids else 1

    moe_sd_alias, num_classes = alias_classifier_to_head(
        moe_sd, d_model=hp["d_model"], num_experts=num_experts, num_modalities=2
    )
    hp["num_classes"] = num_classes
    print(f"[info] inferred: d_model={hp['d_model']}  seq_len={hp['seq_len']}  "
        f"num_layers={hp['num_layers']}  num_classes={hp['num_classes']}  experts={num_experts}")

    # 3) 构建 fusion + MoE
    fusion = FusionTransformer(
        num_modalities=hp["num_modalities"],
        d_model=hp["d_model"],
        num_layers=hp["num_layers"],
        num_heads=hp["num_heads"],
        seq_len=hp["seq_len"],
        num_classes=hp["num_classes"],
    )
    moe = DiME(
        num_modalities=2,
        fusion_model=fusion,
        fusion_sparse=True,
        hidden_dim=hp["d_model"],
        hidden_dim_rw=256,
        num_layer_rw=2,
        temperature_rw=1,
    )

    # 4) 用带别名的 state_dict 加载（就能把 head.* 也对上）
    def drop_keys(sd, prefixes=("reweight.",)):
        removed = []
        kept = {}
        for k, v in sd.items():
            if k.startswith(prefixes):
                removed.append(k)
            else:
                kept[k] = v
        print(f"[prune] drop {len(removed)} keys starting with {prefixes}: "
            f"{removed[:4]}{' ...' if len(removed) > 4 else ''}")
        return kept

    moe_sd_clean = drop_keys(moe_sd_alias, prefixes=("reweight.",))
    missing, unexpected = moe.load_state_dict(moe_sd_clean, strict=False)
    print("[moe.load] missing:", missing)
    print("[moe.load] unexpected:", unexpected)
    moe.eval().to(device)


    # 3) 构建并加载两个编码器
    # 如果 enc_pack['T'] / ['S'] 是 state_dict：直接 load；如果是 Module：取其 state_dict 再 load
    def build_and_load_enc(mod_sd_or_mod, which):
        # 根据 fusion 输入维度确定 hidden_dim；按你的预处理与模型，hidden_dim = d_model
        enc = BertEncoder(hp["d_model"], patch=True, num_patches=hp["seq_len"]//2).eval().to(device) if which=="T" \
            else CLIPImageEncoder(hp["d_model"], patch=True, num_patches=hp["seq_len"]//2).eval().to(device)
        if isinstance(mod_sd_or_mod, nn.Module):
            enc.load_state_dict(mod_sd_or_mod.state_dict(), strict=False)
        elif isinstance(mod_sd_or_mod, dict):
            enc.load_state_dict(mod_sd_or_mod, strict=False)
        else:
            raise TypeError(f"Unexpected encoder type for {which}: {type(mod_sd_or_mod)}")
        return enc

    enc_T = build_and_load_enc(enc_pack["T"], "T")
    enc_S = build_and_load_enc(enc_pack["S"], "S")

    # 4) 预处理输入并拿到嵌入
    img_tensor, pil_img = load_image_tensor(args.image)
    tok = tokenize_text(args.tweet, args.cot)
    for k in tok: tok[k] = tok[k].to(device)

    with torch.no_grad():
        t_embed = enc_T(tok)                     # [1,P,H]
        s_embed = enc_S(img_tensor.to(device))   # [1,P,H]

    inputs_list = [t_embed, s_embed]
    img_mod_idx = 1

    # 5) 遍历专家，输出热力图
    num_experts = len(moe.interaction_experts)
    print(f"[info] experts={num_experts} | d_model={hp['d_model']} | seq_len={hp['seq_len']} | num_layers={hp['num_layers']} | classes={hp['num_classes']}")
    for e in range(num_experts):
        heat, tgt = occlusion_heatmap_for_expert(
            moe_model=moe, expert_index=e, inputs_embed_list=inputs_list,
            img_encoder=enc_S, img_tensor=img_tensor,
            img_modality_index=img_mod_idx, target_class=args.target,
            window=args.win, stride=args.stride, baseline_mode=args.baseline,
            score_use_softmax=args.use_softmax
        )
        out_png = os.path.join(args.out_dir, f"expert{e:02d}_target{tgt}.png")
        overlay_and_save(pil_img, heat, out_png, alpha=0.45)
        print(f"[saved] {out_png}")

    print("[done] all experts exported.]")

if __name__ == "__main__":
    main()
