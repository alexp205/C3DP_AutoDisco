import argparse
from tools.extractor import run as run_extract
from tools.refscrape import run as run_refscrape
from tools.matpred import run as run_matpred
from tools.discover import run as run_discover

parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=int, default=1)

def main():
    args = parser.parse_args()
    opt = args.opt

    if 1 == opt:
        print("--- RUNNING DATASET EXTRACTION ---")
        run_extract()
    elif 2 == opt:
        print("--- RUNNING DATASET EXPANSION ---")
        run_refscrape()
    elif 3 == opt:
        print("--- RUNNING STRENGTH LEARNING ---")
        run_matpred()
    elif 4 == opt:
        print("--- RUNNING MATERIAL DISCOVERY ---")
        run_discover()

    print()
    print("--- ROUTINE COMPLETED! ---")
    print()

if __name__ == "__main__":
    main()
