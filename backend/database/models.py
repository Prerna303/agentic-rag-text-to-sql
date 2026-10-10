"""
FILE PURPOSE
------------
Defines the demo rental-business schema (DDL) and seeds it with reproducible
random data. This is the ONLY place where the demo tables are written.

TABLES  : customers, branches, vehicles, rental, rating, maintenance
          (FKs: vehicles.branch_id->branches, rental.customer_id->customers,
           rental.vehicle_id->vehicles, rating.rental_id->rental,
           maintenance.vehicle_id->vehicles)
WHY 6 TABLES: with only 4 tables a TOP_K=5 retrieval would return everything and
          RAG would be trivial. Six tables make retrieval meaningful.
INPUT  : counts (configurable), random seed.  OUTPUT: data/database.db
CALLED BY: scripts/init_db.py, tests/conftest.py.  CALLS: sqlite3 (stdlib).
EXAMPLE: create_demo_database("data/database.db", customers=100, vehicles=50)
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Jaipur"]
CATEGORIES = ["SUV", "Sedan", "Hatchback", "Luxury", "Van"]
MANUFACTURERS = ["Toyota", "Hyundai", "Honda", "Tata", "Mahindra", "Suzuki", "Kia"]
BRANCH_CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad"]

DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT,
    city TEXT, registration_date TEXT);
CREATE TABLE branches (
    branch_id INTEGER PRIMARY KEY, branch_name TEXT NOT NULL, city TEXT, opened_year INTEGER);
CREATE TABLE vehicles (
    vehicle_id INTEGER PRIMARY KEY, model TEXT NOT NULL, manufacturer TEXT, category TEXT,
    price_per_day REAL, branch_id INTEGER REFERENCES branches(branch_id));
CREATE TABLE rental (
    rental_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    vehicle_id INTEGER NOT NULL REFERENCES vehicles(vehicle_id),
    rental_date TEXT, return_date TEXT, amount REAL);
CREATE TABLE rating (
    rating_id INTEGER PRIMARY KEY,
    rental_id INTEGER NOT NULL REFERENCES rental(rental_id),
    rating INTEGER, review_text TEXT);
CREATE TABLE maintenance (
    maintenance_id INTEGER PRIMARY KEY,
    vehicle_id INTEGER NOT NULL REFERENCES vehicles(vehicle_id),
    service_date TEXT, cost REAL, description TEXT);
"""

FIRST = ["Aarav", "Vivaan", "Aditya", "Ananya", "Diya", "Ishaan", "Kavya", "Rohan", "Sneha", "Priya",
         "Arjun", "Meera", "Karan", "Neha", "Rahul", "Pooja", "Sanjay", "Anjali", "Vikram", "Riya"]
LAST = ["Sharma", "Verma", "Reddy", "Patel", "Singh", "Iyer", "Das", "Nair", "Gupta", "Khan"]
MODELS = {"SUV": ["Creta", "Fortuner", "XUV700", "Seltos", "Nexon"],
          "Sedan": ["City", "Verna", "Ciaz", "Camry"],
          "Hatchback": ["i20", "Swift", "Baleno", "Altroz"],
          "Luxury": ["Innova Crysta", "Accord", "Alcazar"],
          "Van": ["Ertiga", "Carens", "Marazzo"]}
PRICE = {"SUV": (2500, 4500), "Sedan": (1800, 3200), "Hatchback": (1000, 2000),
         "Luxury": (4500, 8000), "Van": (2200, 3800)}
REVIEWS = {1: "Terrible experience", 2: "Below expectations", 3: "Average trip",
           4: "Good vehicle", 5: "Excellent, would rent again"}
SERVICES = ["Oil change", "Brake service", "Tyre replacement", "Engine check", "General service"]


def create_demo_database(path: str | Path, customers: int = 100, vehicles: int = 50,
                         rentals: int = 500, ratings: int = 300, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rnd = random.Random(seed)
    con = sqlite3.connect(path)
    con.executescript(DDL)

    con.executemany("INSERT INTO branches VALUES (?,?,?,?)",
                    [(i + 1, f"{c} Central", c, rnd.randint(2010, 2020)) for i, c in enumerate(BRANCH_CITIES)])
    base = date(2022, 1, 1)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", [
        (i, (n := f"{rnd.choice(FIRST)} {rnd.choice(LAST)}"), f"user{i}@example.com",
         rnd.choice(CITIES), (base + timedelta(days=rnd.randint(0, 1000))).isoformat())
        for i in range(1, customers + 1)])
    vrows = []
    for i in range(1, vehicles + 1):
        cat = CATEGORIES[(i - 1) % len(CATEGORIES)]
        lo, hi = PRICE[cat]
        vrows.append((i, rnd.choice(MODELS[cat]), rnd.choice(MANUFACTURERS), cat,
                      float(rnd.randrange(lo, hi, 100)), rnd.randint(1, len(BRANCH_CITIES))))
    con.executemany("INSERT INTO vehicles VALUES (?,?,?,?,?,?)", vrows)
    price = {r[0]: r[4] for r in vrows}
    rrows = []
    for i in range(1, rentals + 1):
        vid = rnd.randint(1, max(1, int(vehicles * 0.9)))   # last ~10% of vehicles are never rented
        start = date(2023, 1, 1) + timedelta(days=rnd.randint(0, 700))
        days = rnd.randint(1, 10)
        rrows.append((i, rnd.randint(1, max(1, int(customers * 0.85))), vid, start.isoformat(),
                      (start + timedelta(days=days)).isoformat(), price[vid] * days))
    con.executemany("INSERT INTO rental VALUES (?,?,?,?,?,?)", rrows)
    rated = rnd.sample(range(1, rentals + 1), min(ratings, rentals))
    con.executemany("INSERT INTO rating VALUES (?,?,?,?)", [
        (i + 1, rid, (r := rnd.choices([1, 2, 3, 4, 5], [1, 2, 4, 6, 5])[0]), REVIEWS[r])
        for i, rid in enumerate(rated)])
    mvehicles = rnd.sample(range(1, vehicles + 1), int(vehicles * 0.7))
    mrows, mid = [], 1
    for vid in mvehicles:
        for _ in range(rnd.randint(1, 3)):
            mrows.append((mid, vid, (date(2023, 1, 1) + timedelta(days=rnd.randint(0, 700))).isoformat(),
                          float(rnd.randrange(500, 6000, 100)), rnd.choice(SERVICES)))
            mid += 1
    con.executemany("INSERT INTO maintenance VALUES (?,?,?,?,?)", mrows)
    con.commit()
    con.close()
    return path
