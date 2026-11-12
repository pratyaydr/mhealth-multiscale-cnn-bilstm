"""Command-line entry point. Training imports stay lazy for lightweight audits."""

import argparse
from pathlib import Path

from .config import load_config
from .data import load_split, prepare
from .evaluation import audit_saved_results, export_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="mHealth activity recognition experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        if command == "prepare":
            sub.add_argument("--data")
            sub.add_argument("--output")
        else:
            sub.add_argument("--output")
    audit = commands.add_parser("audit")
    audit.add_argument("--evidence", default="evidence/original")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="test")
    args = parser.parse_args()
    if args.command == "audit":
        print(audit_saved_results(args.evidence).to_string(index=False))
    elif args.command == "prepare":
        config = load_config(args.config)
        if args.data:
            config["data_path"] = args.data
        if args.output:
            config["output_dir"] = args.output
        print(prepare(config).to_string(index=False))
    elif args.command == "train":
        from .training import train

        config = load_config(args.config)
        if args.output:
            config["output_dir"] = args.output
        print(train(config))
    elif args.command == "evaluate":
        import tensorflow as tf

        run = Path(args.run)
        config = load_config(run / "config.json")
        x, y, activities = load_split(
            Path(config["data_dir"]) / f"{args.split}.npz", config["removed_activity"]
        )
        destination = run / f"evaluation_{args.split}"
        if destination.exists():
            raise FileExistsError(f"Evaluation already exists: {destination}")
        model = tf.keras.models.load_model(run / "model.keras", compile=False)
        probabilities = model.predict(x, batch_size=config["batch_size"], verbose=0)
        print(export_evaluation(y, probabilities, activities, destination))


if __name__ == "__main__":
    main()
