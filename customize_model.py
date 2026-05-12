import argparse
from pathlib import Path

from diffusers import AutoModel
from open_flamingo import create_model_and_transforms

def parse():
    args = argparse.ArgumentParser()
    args.add_argument("--vision_path",type=str,default=None,help="path to vision model")
    args.add_argument("--lang_path",type=str,default=None,help="path to language model")
    args.add_argument("--n_layers",type=int,default=1,help="cross attention n layers")
    args.add_argument("--output_path",type=str,default=None,help="path to output model")

    return args.parse_args()

def main():
    args = parse()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14", # dummy path
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.lang_path,
        tokenizer_path=args.lang_path,
        cross_attn_every_n_layers=args.n_layers,
    )
    # force load vision encoder
    vision_encoder = AutoModel.from_pretrained(args.vision_path,trust_remote_code=True)
    model.vision_encoder =  vision_encoder

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    image_processor.save_pretrained(output_path)


if __name__ == "__main__":
    main()
