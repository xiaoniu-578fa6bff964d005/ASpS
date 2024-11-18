from datasets import load_dataset
import os


def exp_debug_cut(ds):
    if os.environ.get("EXP_DEBUG", None) == "1":
        ds = ds.select(range(0, 2))
    if os.environ.get("EXP_DEBUG", None) == "2":
        ds = ds.select(range(0, 10))
    if os.environ.get("EXP_DEBUG", None) == "3":
        ds = ds.select(range(0, 100))
    return ds


def get_translation_ds():
    wmt16 = load_dataset("wmt16", "ro-en")
    ds = wmt16["test"]
    ds = ds.select(range(0, 1000))
    ds = exp_debug_cut(ds)
    ds = ds.flatten()

    def create_prompt(d, idx):
        s = {}
        s["idx"] = idx
        s["prompt"] = "Translate from English to Romanian: " + d["translation.en"]
        return s

    ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
    return ds


def get_summarization_ds():
    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["test"]
    ds = ds.filter(lambda x: len(x["article"]) < 3000).select(range(0, 1000))
    ds = exp_debug_cut(ds)

    def create_prompt(d, idx):
        s = {}
        s["idx"] = idx
        s["prompt"] = d["article"] + "\nTL;DR:\n"
        return s

    ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
    return ds


def get_oeg_ds():
    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["test"]
    ds = ds.select(range(0, 1000))
    ds = exp_debug_cut(ds)

    def smart_truncate(content, length):
        if len(content) <= length:
            return content
        else:
            return " ".join(content[: length + 1].split(" ")[0:-1])

    def create_prompt(d, idx):
        s = {}
        s["idx"] = idx
        s["prompt"] = smart_truncate(d["article"], length=100)
        return s

    ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
    return ds
