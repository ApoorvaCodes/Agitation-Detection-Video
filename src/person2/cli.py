"""Run evidence extraction and build prototypes from explicitly labelled chunks."""
import argparse
import json
from pathlib import Path

from person1.io import load_perception
from person2.config import Person2Config
from person2.embeddings import RGBHistogramEncoder, TemporalPoseEncoder, encoder_metadata, load_video_encoder
from person2.experiments import run_experiments
from person2.io import load_prototypes, load_result, save_prototypes, save_result
from person2.pipeline import process_perception
from person2.prototypes import build_prototypes
from cmai.taxonomy import require_initial_items


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config", help="JSON object of Person2Config fields")
    run.add_argument("--prototypes")
    video = run.add_mutually_exclusive_group()
    video.add_argument("--video", help="Explicit local source video for RGB crop histogram baseline")
    video.add_argument("--video-encoder", help="JSON local encoder factory, kwargs, and model identity")
    run.add_argument("--pose-representation", choices=("stats", "time-bins"), default="stats")
    run.add_argument("--temporal-bins", type=int, default=4)
    run.add_argument("--metadata-output", help="Encoder identity sidecar; defaults to OUTPUT.encoders.json")
    build = sub.add_parser("build-prototypes")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--version", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("path")
    experiment = sub.add_parser("experiment")
    experiment.add_argument("--manifest", help="Explicitly annotated subject/session-disjoint dataset JSON")
    experiment.add_argument("--output-dir", required=True)
    experiment.add_argument("--config")
    experiment.add_argument("--thresholds", type=float, nargs="+", default=[.6, .7, .8, .9])
    experiment.add_argument("--temporal-bins", type=int, default=4)
    args = parser.parse_args()
    if args.command == "run":
        metadata_path = Path(args.metadata_output or f"{args.output}.encoders.json")
        if metadata_path.resolve() == Path(args.output).resolve():
            raise ValueError("metadata output must differ from the contract output")
        config = Person2Config(**json.loads(Path(args.config).read_text())) if args.config else Person2Config()
        pose = TemporalPoseEncoder(config.window_seconds, args.temporal_bins) if args.pose_representation == "time-bins" else None
        video_encoder = load_video_encoder(args.video_encoder) if args.video_encoder else RGBHistogramEncoder(args.video) if args.video else None
        result = process_perception(load_perception(args.input), config,
                                    load_prototypes(args.prototypes) if args.prototypes else None,
                                    video_encoder, pose)
        save_result(result, args.output)
        metadata_path.write_text(json.dumps(encoder_metadata(result, pose, video_encoder), indent=2, allow_nan=False) + "\n")
        print(f"saved persons={len(result.persons)} chunks={sum(len(p.chunks) for p in result.persons)}")
    elif args.command == "build-prototypes":
        manifest_path = Path(args.manifest)
        examples = []
        for item in json.loads(manifest_path.read_text()):
            if item.get("split") != "train":
                raise ValueError("prototype builder accepts explicitly labelled training examples only")
            require_initial_items([item["behaviour"]])
            result = load_result(manifest_path.parent / item["result"])
            matches = [c for p in result.persons for c in p.chunks if c.chunk_id == item["chunk_id"]]
            if len(matches) != 1 or matches[0].status in {"low_quality", "insufficient_evidence"}:
                raise ValueError(f"chunk missing, ambiguous, or unusable: {item['chunk_id']}")
            examples.append((item["behaviour"], matches[0].fused_embedding))
        if not examples:
            raise ValueError("manifest must contain labelled examples")
        save_prototypes(build_prototypes(examples, args.version), args.output)
        print(f"saved prototypes from {len(examples)} labelled chunks")
    elif args.command == "experiment":
        config = Person2Config(**json.loads(Path(args.config).read_text())) if args.config else Person2Config()
        report = run_experiments(args.manifest, args.output_dir, config, args.thresholds, args.temporal_bins)
        print(f"experiment status={report['status']} report={Path(args.output_dir) / 'report.json'}")
    else:
        result = load_result(args.path)
        print(f"valid schema={result.schema_version} persons={len(result.persons)}")


if __name__ == "__main__":
    main()
