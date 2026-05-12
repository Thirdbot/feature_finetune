import argparse
import shutil
from pathlib import Path

from open_flamingo import create_model_and_transforms
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_path", type=str, default=None, help="path to vision model")
    parser.add_argument("--lang_path", type=str, default=None, help="path to language model")
    parser.add_argument("--n_layers", type=int, default=1, help="cross attention n layers")
    parser.add_argument("--output_path", type=str, required=True, help="path to save registered model")
    return parser.parse_args()


class FlamingoConfig(PretrainedConfig):
    model_type = "custom"

    def __init__(
        self,
        vision_path=None,
        lang_path=None,
        n_layers=1,
        vision_config=None,
        text_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_path = vision_path
        self.lang_path = lang_path
        self.n_layers = n_layers
        self.vision_config = vision_config
        self.text_config = text_config


# Backward-compatible alias for the previous typo.
FlamingoConnfig = FlamingoConfig


class FlamingoModel(PreTrainedModel):
    config_class = FlamingoConfig

    def __init__(self, config):
        super().__init__(config)
        if not config.vision_path:
            raise ValueError("FlamingoConfig.vision_path is required to build the model.")
        if not config.lang_path:
            raise ValueError("FlamingoConfig.lang_path is required to build the model.")

        model, image_processor, tokenizer = build_flamingo_model(
            vision_path=config.vision_path,
            lang_path=config.lang_path,
            n_layers=config.n_layers,
        )
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.model, "prepare_inputs_for_generation"):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)


def build_flamingo_model(vision_path, lang_path, n_layers):
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=lang_path,
        tokenizer_path=lang_path,
        cross_attn_every_n_layers=n_layers,
    )
    model.vision_encoder = AutoModel.from_pretrained(vision_path, trust_remote_code=True)
    return model, image_processor, tokenizer


def register_model():
    AutoConfig.register(FlamingoConfig.model_type, FlamingoConfig, exist_ok=True)
    AutoModelForCausalLM.register(FlamingoConfig, FlamingoModel, exist_ok=True)


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    register_model()
    FlamingoConfig.register_for_auto_class()
    FlamingoModel.register_for_auto_class("AutoModelForCausalLM")

    config = FlamingoConfig(
        vision_path=args.vision_path,
        lang_path=args.lang_path,
        n_layers=args.n_layers,
    )
    config.auto_map = {
        "AutoConfig": "register.FlamingoConfig",
        "AutoModelForCausalLM": "register.FlamingoModel",
    }
    model = FlamingoModel(config)
    model.save_pretrained(output_path)
    model.tokenizer.save_pretrained(output_path)
    model.image_processor.save_pretrained(output_path)
    source_file = Path(__file__).resolve()
    target_file = output_path.resolve() / source_file.name
    if source_file != target_file:
        shutil.copyfile(source_file, target_file)

    print(f"registered model saved to {output_path}")


register_model()


if __name__ == "__main__":
    main()
