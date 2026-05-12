import argparse
from pathlib import Path

import torch
from torchview import draw_graph


def parse():
    parser = argparse.ArgumentParser("View architectures of desired model")

    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    parser.add_argument("--name", type=str, default="model", help="Name of the model")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--component", action="store_true", help="Trace a standalone component like the VL3 encoder")
    parser.add_argument("--expand", action="store_true", help="Expands arcs")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Single device used for tracing. Do not use accelerate offload with torchview.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Dummy token sequence length for tracing",
    )
    return parser.parse_args()


def load_model(args,is_component=False):
    if not is_component:
        try:
                from unsloth import FastLanguageModel

                model, _ = FastLanguageModel.from_pretrained(
                    model_name=args.model_path,
                )
                return model
        except Exception as exc:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
            )
            return model
    else:
        from transformers import AutoModel

        model =  AutoModel.from_pretrained(args.model_path,trust_remote_code=True)
        return model


def make_dummy_input(args, model):
    if args.component:
        # vision component only
        patch_size = getattr(model.config, "patch_size", 14)
        num_channels = getattr(model.config, "num_channels", 3)
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

        grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
        merge_sizes = torch.tensor([1], dtype=torch.long)
        num_patches = int(grid_sizes.prod(dim=1).sum().item())
        pixel_values = torch.zeros(
            (num_patches, num_channels, patch_size, patch_size),
            dtype=dtype,
        )
        return {
            "pixel_values": pixel_values,
            "grid_sizes": grid_sizes,
            "merge_sizes": merge_sizes,
        }

    example_batchsize = 1
    return torch.zeros((example_batchsize, args.seq_len), dtype=torch.long)

def main():
    args = parse()
    output_path = Path(args.output_path or ".")
    output_path.mkdir(parents=True, exist_ok=True)

    model = load_model(args,args.component)
    dummy_input = make_dummy_input(args, model)

    graph = draw_graph(
        model=model,
        graph_name=args.name,
        input_data=dummy_input,
        filename=args.name,
        directory=str(output_path),
        device=args.device,
        expand_nested=args.expand,
        save_graph=True,
    )
    graph.visual_graph.render(
        filename=args.name,
        directory=str(output_path),
        format="png",
        cleanup=True,
    )

if __name__ == "__main__":
    main()
