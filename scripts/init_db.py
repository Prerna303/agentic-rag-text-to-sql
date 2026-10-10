"""
FILE PURPOSE: creates data/database.db (demo rental database) with reproducible data.
USAGE: python scripts/init_db.py [--customers 100 --vehicles 50 --rentals 500 --ratings 300 --seed 42]
CALLED BY: humans / README.  CALLS: backend/database/models.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import PROJECT_ROOT  # noqa: E402
from backend.database.models import create_demo_database  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=100)
    ap.add_argument("--vehicles", type=int, default=50)
    ap.add_argument("--rentals", type=int, default=500)
    ap.add_argument("--ratings", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    p = create_demo_database(PROJECT_ROOT / "data" / "database.db", a.customers, a.vehicles,
                             a.rentals, a.ratings, a.seed)
    print(f"Created {p}")
