import argparse, json
from pathlib import Path
from person1 import Person1Config, process_video, save_perception, save_perception_jsonl, save_perception_npz
from person1.overlay import render_overlay
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True,type=Path); p.add_argument("--config",type=Path,default=Path("configs/default.yaml")); p.add_argument("--output",type=Path); p.add_argument("--output-jsonl",type=Path); p.add_argument("--output-npz",type=Path); p.add_argument("--save-overlay",type=Path); args=p.parse_args()
    result=process_video(args.input,Person1Config.from_yaml(args.config) if args.config.exists() else Person1Config())
    if args.output: save_perception(result,args.output)
    if args.output_jsonl: save_perception_jsonl(result,args.output_jsonl)
    if args.output_npz: save_perception_npz(result,args.output_npz)
    if args.save_overlay: render_overlay(args.input,result,args.save_overlay)
    if not any((args.output,args.output_jsonl,args.output_npz,args.save_overlay)): print(json.dumps(result.to_json_dict(),indent=2))
if __name__ == "__main__": main()
