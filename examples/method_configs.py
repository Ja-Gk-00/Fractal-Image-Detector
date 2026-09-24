import argparse

from fractal_detector.presets import write_method_configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write standard/deep TOML configurations for every method"
    )
    parser.add_argument("--output", default="configs/methods")
    parser.add_argument("--train", default="../../data-splits/train.csv")
    parser.add_argument("--validation", default="../../data-splits/validation.parquet")
    parser.add_argument("--test")
    args = parser.parse_args()
    paths = write_method_configs(
        args.output, train=args.train, validation=args.validation, test=args.test
    )
    print(
        f"Wrote {len(paths)} configurations to {args.output}. Relative input paths resolve from that directory."
    )


if __name__ == "__main__":
    main()
