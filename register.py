import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
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
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="maximum shard size passed to save_pretrained",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="save weights as safetensors instead of pytorch_model.bin",
    )
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


def load_config_dict(model_path):
    if model_path is None:
        return None
    return AutoConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()


def clone_shared_tensors(state_dict):
    cloned_state_dict = {}
    seen_storages = set()

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            cloned_state_dict[name] = tensor
            continue

        storage_key = (tensor.device, tensor.untyped_storage().data_ptr())
        if storage_key in seen_storages:
            cloned_state_dict[name] = tensor.clone()
        else:
            seen_storages.add(storage_key)
            cloned_state_dict[name] = tensor

    return cloned_state_dict


def assert_saved(output_path, safe_serialization):
    config_file = output_path / "config.json"
    weight_patterns = ["*.safetensors"] if safe_serialization else ["*.bin"]
    weight_files = [
        weight_file
        for pattern in weight_patterns
        for weight_file in output_path.glob(pattern)
    ]

    if not config_file.exists() or not weight_files:
        found_files = ", ".join(sorted(path.name for path in output_path.iterdir()))
        raise RuntimeError(
            f"save_pretrained did not create a complete model in {output_path}. "
            f"Found: {found_files or 'nothing'}"
        )


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
        vision_config=load_config_dict(args.vision_path),
        text_config=load_config_dict(args.lang_path),
    )
    config.auto_map = {
        "AutoConfig": "register.FlamingoConfig",
        "AutoModelForCausalLM": "register.FlamingoModel",
    }
    model = FlamingoModel(config)
    state_dict = clone_shared_tensors(model.state_dict()) if args.safe_serialization else None
    model.save_pretrained(
        output_path,
        safe_serialization=args.safe_serialization,
        state_dict=state_dict,
        max_shard_size=args.max_shard_size,
    )
    model.tokenizer.save_pretrained(output_path)
    model.image_processor.save_pretrained(output_path)
    source_file = Path(__file__).resolve()
    target_file = output_path.resolve() / source_file.name
    if source_file != target_file:
        shutil.copyfile(source_file, target_file)
    assert_saved(output_path, args.safe_serialization)

    print(f"registered model saved to {output_path}")


register_model()


if __name__ == "__main__":
    main()
