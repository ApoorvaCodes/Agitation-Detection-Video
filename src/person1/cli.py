import argparse
from person1.io import load_perception
def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True); validate=sub.add_parser("validate"); validate.add_argument("path")
    args=parser.parse_args()
    if args.command=="validate": result=load_perception(args.path); print(f"valid schema={result.schema_version} persons={len(result.persons)}")
if __name__=="__main__": main()
