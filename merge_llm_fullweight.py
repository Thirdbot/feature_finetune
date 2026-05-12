from unsloth import FastLanguageModel
from peft import PeftModel
import argparse
import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "Unsloth requires a CUDA GPU. Download the model first with "
        "`python download_model.py`, then run this script on a GPU machine."
    )

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,help="Path to the model")
    parser.add_argument("--seq_len",type=int,default=2048,help="The sequence length")
    parser.add_argument("--adapter_path",type=str,help="Path to the adapter")
    parser.add_argument("--output_dir",type=str,help="Path to the output directory")
    return parser.parse_args()

def main():
    args = parse()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.seq_len,
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # load additional weights
    model = PeftModel.from_pretrained(
    model,
    args.adapter_path,
    )
    # merge model weights back
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
