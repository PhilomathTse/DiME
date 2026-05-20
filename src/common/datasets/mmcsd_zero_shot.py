import os
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer, BertModel,CLIPModel,BertConfig

from src.common.modules.common import  MLP, PatchEmbeddings
#VGG11Slim, Linear,
from src.common.utils import get_modality_combinations

topic_text_map = {
    "Mainland of China": "The stance on Mainland of China is:",
    "Taiwan of China": "The stance on Taiwan of China is:",
    "Donald Trump": "The stance on Donald Trump is:",
    "Joe Biden": "The stance on Joe Biden is:",
    "Merger and acquisition between Disney and 21st Century Fox.": "The stance on Merger and acquisition between Disney and 21st Century Fox is:",
    "Merger and acquisition between Aetna and Humana.":"The stance on Merger and acquisition between Aetna and Humana. is:",
}

BASE_DIR = "/home/zbw/project/SILO-MM/SILO-SD/Dataset"

DATA_PATHS = {
    "CQ": {
        "target": "CQ",
        "dt_dir": f"{BASE_DIR}/COVID-CQ/zero-shot/CQ",
        "img_root": f"{BASE_DIR}/COVID-CQ/images",
    },
    "DT": {
        "target": "DT",
        "dt_dir": f"{BASE_DIR}/2020-US-Presidential-Election/zero-shot/DT",
        "img_root": f"{BASE_DIR}/2020-US-Presidential-Election/images",
    },
    "JB": {
        "target": "JB",
        "dt_dir": f"{BASE_DIR}/2020-US-Presidential-Election/zero-shot/JB",
        "img_root": f"{BASE_DIR}/2020-US-Presidential-Election/images",
    },
    "TOC": {
        "target": "TOC",
        "dt_dir": f"{BASE_DIR}/TW-Question/zero-shot/TOC",
        "img_root": f"{BASE_DIR}/TW-Question/images",
    },
    "MOC": {
        "target": "MOC",
        "dt_dir": f"{BASE_DIR}/TW-Question/zero-shot/MOC",
        "img_root": f"{BASE_DIR}/TW-Question/images",
    },
    "AET_HUM": {
        "target": "AET_HUM",
        "dt_dir": f"{BASE_DIR}/WT-WT/zero-shot/AET_HUM",
        "img_root": f"{BASE_DIR}/WT-WT/images",
    },
    "ANTM_CI": {
        "target": "ANTM_CI",
        "dt_dir": f"{BASE_DIR}/WT-WT/zero-shot/ANTM_CI",
        "img_root": f"{BASE_DIR}/WT-WT/images",
    },
    "CI_ESRX": {
        "target": "CI_ESRX",
        "dt_dir": f"{BASE_DIR}/WT-WT/zero-shot/CI_ESRX",
        "img_root": f"{BASE_DIR}/WT-WT/images",
    },
    "CSV_AET": {
        "target": "CSV_AET",
        "dt_dir": f"{BASE_DIR}/WT-WT/zero-shot/CSV_AET",
        "img_root": f"{BASE_DIR}/WT-WT/images",
    },
    "FOXA_DIS": {
        "target": "FOXA_DIS",
        "dt_dir": f"{BASE_DIR}/WT-WT/zero-shot/FOXA_DIS",
        "img_root": f"{BASE_DIR}/WT-WT/images",
    },
    "UKR": {
        "target": "UKR",
        "dt_dir": f"{BASE_DIR}/RU-Conflict/zero-shot/UKR",
        "img_root": f"{BASE_DIR}/RU-Conflict/images",
    },
    "RUS": {
        "target": "RUS",
        "dt_dir": f"{BASE_DIR}/RU-Conflict/zero-shot/RUS",
        "img_root": f"{BASE_DIR}/RU-Conflict/images",
    },
}

# DT_DIR = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/COVID-CQ/in-target/CQ"
# IMG_ROOT = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/COVID-CQ//images"
# DT_DIR = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/2020-US-Presidential-Election/in-target/JB"
# IMG_ROOT = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/2020-US-Presidential-Election/images"
# DT_DIR = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/TW-Question/in-target/TOC"
# IMG_ROOT = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/TW-Question/images"
# DT_DIR = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/WT-WT/in-target/AET_HUM"
# IMG_ROOT = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/WT-WT/images"
# DT_DIR = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/RU-Conflict/in-target/UKR"
# IMG_ROOT = "/home/zbw/project/SILO-MM/SILO-SD/Dataset/RU-Conflict/images"

    
class CLIPImageEncoder(nn.Module): 
    """Trainable CLIP vision encoder projecting to hidden_dim.""" 
    def __init__(self, hidden_dim, num_layers_enc=1, patch=True, num_patches=16): 
        super().__init__() 
        self.clip = CLIPModel.from_pretrained( "openai/clip-vit-base-patch32", local_files_only=True ) 
        vision_width = self.clip.config.vision_config.hidden_size 
        mlp = MLP(vision_width, hidden_dim, hidden_dim, num_layers_enc) 
        if patch: 
            self.proj = nn.Sequential( mlp, PatchEmbeddings( feature_size=hidden_dim, num_patches=num_patches, embed_dim=hidden_dim, ), ) 
        else: 
            self.proj = mlp 
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, pixel_values): 
        outputs = self.clip.vision_model(pixel_values=pixel_values) 
        pooled = outputs.pooler_output 
        return self.proj(pooled)


def _read_splits(args):
    DT_DIR  = DATA_PATHS[args.original_target]["dt_dir"]

    paths = {
        "train": os.path.join(DT_DIR, "train.csv"),
        "valid": os.path.join(DT_DIR, "valid.csv"),
        "test": os.path.join(DT_DIR, "test.csv"),
    }
    dfs = {}
    need_cols = [
        "tweet_id",
        "tweet_text",
        "stance_target",
        "stance_label",
        "tweet_image",
        "gpt4v_cot_response",
    ]
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[mmcsd] CSV not found: {p}")
        df = pd.read_csv(p)
        print(f"[mmcsd] {k} targets:", df["stance_target"].unique())
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise KeyError(f"[mmcsd] CSV {k} missing columns: {missing}")
        dfs[k] = df
    return dfs


def _build_label_map(all_df):
    uniq = sorted(all_df["stance_label"].astype(str).unique().tolist())
    lab2idx = {lab: i for i, lab in enumerate(uniq)}
    return lab2idx, {i: lab for lab, i in lab2idx.items()}

def _image_to_numpy(abs_path, img_size=(224,224), normalize=True):
    # tfs = [Resize(img_size), ToTensor()]
    # if normalize:
    #     tfs.append(Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    # transform = Compose(tfs)#@Zy

    tfs = [
        Resize(224, interpolation=InterpolationMode.BICUBIC),  
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ]
    transform = Compose(tfs)
    if not abs_path or not os.path.isfile(abs_path):
        return np.zeros((3, img_size[0], img_size[1]), dtype=np.float32), False

    try:
        im = Image.open(abs_path)
        if (im.mode == "P" and "transparency" in im.info) or im.mode == "RGBA":
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        else:
            im = im.convert("RGB")

        ten = transform(im)
        return ten.numpy().astype(np.float32), True
    except Exception:
        return np.zeros((3, img_size[0], img_size[1]), dtype=np.float32), False

# def _image_to_numpy(abs_path, img_size=(224, 224), normalize=True):
#     tfs = [Resize(img_size), ToTensor()]
#     if normalize:
#         tfs.append(
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         )
#     transform = Compose(tfs)

#     if (abs_path is None) or (not os.path.isfile(abs_path)):
#         c, h, w = 3, img_size[0], img_size[1]
#         return np.zeros((c, h, w), dtype=np.float32), False

#     try:
#         im = Image.open(abs_path).convert("RGB")
#         ten = transform(im)
#         return ten.numpy().astype(np.float32), True
#     except Exception:
#         c, h, w = 3, img_size[0], img_size[1]
#         return np.zeros((c, h, w), dtype=np.float32), False


class BertEncoder(nn.Module):
    """Trainable BERT encoder projecting to hidden_dim."""
    def __init__(self, hidden_dim, num_layers_enc=1, patch=True, num_patches=16):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        mlp = MLP(768, hidden_dim, hidden_dim, num_layers_enc)
        if patch:
            self.proj = nn.Sequential(
                mlp,
                PatchEmbeddings(
                    feature_size=hidden_dim,
                    num_patches=num_patches,
                    embed_dim=hidden_dim,),
            )
        else:
            self.proj = mlp

    def forward(self, inputs):
        outputs = self.bert(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        if outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.proj(pooled)
# class BertEncoder(nn.Module):
#     def __init__(self, hidden_dim, num_layers_enc=1, patch=True, num_patches=16):
#         super().__init__()
#         # 拿到 tokenizer 的 vocab_size（离线可用；拿不到就 30522）
#         try:
#             _tok = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
#             _vocab_size = _tok.vocab_size
#         except Exception:
#             _vocab_size = 30522

#         cfg = BertConfig(
#             vocab_size=_vocab_size,      
#             hidden_size=768,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=3072,
#             hidden_dropout_prob=0.1,
#             attention_probs_dropout_prob=0.1,
#         )
#         self.bert = BertModel(cfg)      

#         mlp = MLP(768, hidden_dim, hidden_dim, num_layers_enc)
#         if patch:
#             self.proj = nn.Sequential(
#                 mlp,
#                 PatchEmbeddings(feature_size=hidden_dim, num_patches=num_patches, embed_dim=hidden_dim),
#             )
#         else:
#             self.proj = mlp

#     def forward(self, inputs):
#         outputs = self.bert(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
#         pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
#         return self.proj(pooled)

def load_and_preprocess_data_mmcsd(args):
    """Load MMCSD dataset with trainable BERT text encoder."""
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    IMG_ROOT = DATA_PATHS[args.original_target]["img_root"]
    # 1) Read splits and merge
    dfs = _read_splits(args)
    all_df = pd.concat([dfs["train"], dfs["valid"], dfs["test"]], axis=0)
    all_df = all_df.drop_duplicates(subset=["tweet_id"], keep="first").reset_index(drop=True)
    # 2) Labels
    label2idx, idx2label = _build_label_map(all_df)
    labels = np.array(
        [label2idx[str(x)] for x in all_df["stance_label"].astype(str)],
        dtype=np.int64,
    )
    n_labels = len(label2idx)

    # 3) Build texts with target and CoT
    # texts = (
    #     all_df["stance_target"].astype(str).fillna("")
    #     + " [SEP] "
    #     + all_df["tweet_text"].astype(str).fillna("")
    #     + " [SEP] "
    #     + all_df["gpt4v_cot_response"].astype(str).fillna("")
    # ).tolist()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    sep = tokenizer.sep_token or "[SEP]"
    texts = (
        all_df["tweet_text"].astype(str).fillna("")
        + f" {sep} "
        + all_df["gpt4v_cot_response"].astype(str).fillna("")
    ).tolist()

    
    enc = tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors="pt",)
    #@Zy: add topic embeddings
    topic_texts = [
        topic_text_map.get(t, f"The stance on {t} is:")
        for t in all_df["stance_target"].astype(str).fillna("").tolist()
    ]
    enc_topic = tokenizer(topic_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    topic_input_ids = enc_topic["input_ids"].numpy().astype(np.int64)
    topic_attention_mask = enc_topic["attention_mask"].numpy().astype(np.int64)
    
    
    input_ids = enc["input_ids"].numpy().astype(np.int64)
    attention_mask = enc["attention_mask"].numpy().astype(np.int64)
    #@Zy: concatenate topic embeddings
    text_data = [
        {
            "input_ids": input_ids[i],
            "attention_mask": attention_mask[i],
            "topic_input_ids": topic_input_ids[i],
            "topic_attention_mask": topic_attention_mask[i],
        }
        for i in range(len(texts))
    ]    
    # text_data = [
    #     {"input_ids": input_ids[i], "attention_mask": attention_mask[i]}
    #     for i in range(len(texts))
    # ]

    # 4) Images
    img_list = []
    img_ok = []
    for rel in all_df["tweet_image"].astype(str).tolist():
        basename = os.path.basename(rel)
        abs_path = os.path.join(IMG_ROOT, basename)
        arr, ok = _image_to_numpy(abs_path, img_size=(224, 224), normalize=True)
        img_list.append(arr)
        img_ok.append(ok)

    screenshot = np.stack(img_list, axis=0).astype(np.float32)

    # 5) Observed modality matrix (N x 2 -> text, screenshot)
    N = len(all_df)
    observed_idx_arr = np.zeros((N, 2), dtype=bool)
    observed_idx_arr[:, 0] = True  # text always available
    observed_idx_arr[:, 1] = np.array(img_ok, bool)

    # 6) Modality combinations
    combination_to_index = get_modality_combinations(args.modality)
    mc_tokens = []
    for i in range(N):
        comb = ""
        if observed_idx_arr[i, 0]:
            comb += "T"
        if observed_idx_arr[i, 1]:
            comb += "S"
        mc_tokens.append("".join(sorted(set(comb))) if comb else "")
    modality_comb = [
        combination_to_index[c] if c in combination_to_index else -1 for c in mc_tokens
    ]

    data_dict = {
        "T": text_data,
        "S": screenshot,
        "modality_comb": modality_comb,
    }

    # 7) Split indices
    id_to_idx = {tid: i for i, tid in enumerate(all_df["tweet_id"].tolist())}

    def _map(df_split):
        return [id_to_idx[tid] for tid in df_split["tweet_id"].tolist() if tid in id_to_idx]

    train_idxs = _map(dfs["train"])
    valid_idxs = _map(dfs["valid"])
    test_idxs = _map(dfs["test"])

    # 8) Filter samples missing all modalities
    def _all_missing(i):
        return (not observed_idx_arr[i, 0]) and (not observed_idx_arr[i, 1])

    train_idxs = [i for i in train_idxs if not _all_missing(i)]
    valid_idxs = [i for i in valid_idxs if not _all_missing(i)]
    test_idxs = [i for i in test_idxs if not _all_missing(i)]
    
    # 9) Encoders and input dimensions#@Zy
    HIDDEN = getattr(args, "hidden_dim", 128)
    encoder_dict = {}
    if getattr(args, "patch", True):
        encoder_dict["T"] = BertEncoder(
            HIDDEN,
            num_layers_enc=getattr(args, "num_layers_enc", 1),
            patch=True,
            num_patches=getattr(args, "num_patches", 16),
        ).to(device)
        encoder_dict["S"] = CLIPImageEncoder(HIDDEN, num_layers_enc=getattr(args, "num_layers_enc", 1), patch=True, num_patches=getattr(args, "num_patches", 16)).to(device) 
    else:
        encoder_dict["T"] = BertEncoder(
            HIDDEN, num_layers_enc=getattr(args, "num_layers_enc", 1), patch=False
        ).to(device)
        encoder_dict["S"] = CLIPImageEncoder(HIDDEN, num_layers_enc=getattr(args, "num_layers_enc", 1), patch=True, num_patches=getattr(args, "num_patches", 16)).to(device) 
    
    # 9) Encoders and input dimensions
    # HIDDEN = getattr(args, "hidden_dim", 128)
    # encoder_dict = {}
    # if getattr(args, "patch", True):
    #     encoder_dict["T"] = BertEncoder(
    #         HIDDEN,
    #         num_layers_enc=getattr(args, "num_layers_enc", 1),
    #         patch=True,
    #         num_patches=getattr(args, "num_patches", 16),
    #     ).to(device)
    #     encoder_dict["S"] = torch.nn.Sequential(
    #         VGG11Slim(1024, dropout=True, dropoutp=0.2, freeze_features=True),
    #         PatchEmbeddings(
    #             feature_size=1024,
    #             num_patches=getattr(args, "num_patches", 16),
    #             embed_dim=HIDDEN,
    #         ),
    #     ).to(device)

    # else:
    #     encoder_dict["T"] = BertEncoder(
    #         HIDDEN, num_layers_enc=getattr(args, "num_layers_enc", 1), patch=False
    #     ).to(device)
    #     encoder_dict["S"] = torch.nn.Sequential(
    #         VGG11Slim(1024, dropout=True, dropoutp=0.2, freeze_features=True),
    #         Linear(1024, HIDDEN, xavier_init=True),
    #     ).to(device)
        
    # encoder_dict["T"].roberta.resize_token_embeddings(len(tokenizer))
    
    input_dims = {"T": HIDDEN, "S": HIDDEN}

    transforms = {}
    masks = {}

    mc_num_to_mc = {v: k for k, v in combination_to_index.items()}
    mc_idx_dict = {
        mc_num_to_mc[m]: list(np.where(np.array(modality_comb) == m)[0])
        for m in set(modality_comb)
        if m != -1
    }

    return (
        data_dict,
        encoder_dict,
        labels,
        train_idxs,
        valid_idxs,
        test_idxs,
        n_labels,
        input_dims,
        transforms,
        masks,
        observed_idx_arr,
        mc_idx_dict,
        mc_num_to_mc,
    )