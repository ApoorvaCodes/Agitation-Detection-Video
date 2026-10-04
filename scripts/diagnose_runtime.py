"""Print the interpreter/model runtime used by the current process."""
import argparse
from person1.perception import runtime_diagnostics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack")
    args = parser.parse_args()
    for key, value in runtime_diagnostics(args.model, args.tracker).items():
        print(f"{key}={value}")

if __name__ == "__main__":
    main()
