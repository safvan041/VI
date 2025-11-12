# vectorlite/maintenance.py
import sys
from vectorlite.storage_memmap import MemmapStorage

def run_compact(db_path: str):
    print(f"Running compaction for {db_path} ...")
    s = MemmapStorage(db_path)
    s.compact()
    print("Compaction finished successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m vectorlite.maintenance <db_path>")
        sys.exit(1)
    run_compact(sys.argv[1])
