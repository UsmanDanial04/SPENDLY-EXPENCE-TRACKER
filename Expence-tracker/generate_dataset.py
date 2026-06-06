import csv
import random
from datetime import datetime, timedelta

random.seed(42)

# Date range: last 12 months
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365)

def random_date():
    delta = END_DATE - START_DATE
    return (START_DATE + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

def rand_id():
    return ''.join([str(random.randint(0, 9)) for _ in range(random.randint(3, 5))])

def rand_num(n=3):
    return ''.join([str(random.randint(0, 9)) for _ in range(n)])

# ── Description templates per category ─────────────────────────────────────
TEMPLATES = {
    "Food": [
        lambda: f"MCDONALDS #{rand_id()}",
        lambda: f"STARBUCKS STORE {rand_num(4)}",
        lambda: f"SUBWAY #{rand_id()}",
        lambda: f"KFC #{rand_id()}",
        lambda: f"PIZZA HUT {rand_num(4)}",
        lambda: f"DOMINOS PIZZA {rand_num(3)}",
        lambda: f"BURGER KING #{rand_id()}",
        lambda: f"DUNKIN #{rand_num(4)}",
        lambda: f"CHIPOTLE {rand_num(4)}",
        lambda: f"TACO BELL #{rand_id()}",
        lambda: f"WENDYS #{rand_id()}",
        lambda: f"PANERA BREAD {rand_num(4)}",
        lambda: f"FIVE GUYS {rand_num(3)}",
        lambda: f"GRUBHUB ORDER {rand_num(6)}",
        lambda: f"DOORDASH *{random.choice(['FOOD','ORDER','DELIV'])} {rand_num(4)}",
        lambda: f"UBER EATS {rand_num(5)}",
        lambda: f"LOCAL BAKERY {rand_num(3)}",
        lambda: f"PRET A MANGER {rand_num(3)}",
        lambda: f"NANDOS #{rand_id()}",
        lambda: f"JUST EAT {rand_num(6)}",
    ],
    "Transport": [
        lambda: f"UBER TRIP {rand_num(4)}",
        lambda: f"LYFT *RIDE {rand_num(5)}",
        lambda: f"SHELL PETROL STN {rand_num(3)}",
        lambda: f"BP FUEL {rand_num(4)}",
        lambda: f"EXXON #{rand_num(4)}",
        lambda: f"CHEVRON {rand_num(4)}",
        lambda: f"METRO TRANSIT {rand_num(5)}",
        lambda: f"CITY BUS PASS {rand_num(3)}",
        lambda: f"NATIONAL RAIL {rand_num(6)}",
        lambda: f"PARKING METER {rand_num(4)}",
        lambda: f"PARKRIGHT {rand_num(5)}",
        lambda: f"ENTERPRISE RENT-A-CAR {rand_num(4)}",
        lambda: f"HERTZ #{rand_num(5)}",
        lambda: f"TESLA SUPERCHARGER {rand_num(4)}",
        lambda: f"TOTAL GAS {rand_num(4)}",
        lambda: f"SUNOCO #{rand_num(4)}",
        lambda: f"AMTRAK TKT {rand_num(6)}",
        lambda: f"GREYHOUND BUS {rand_num(5)}",
        lambda: f"ZIPCAR {rand_num(5)}",
        lambda: f"TOLL PAYMENT {rand_num(6)}",
    ],
    "Housing": [
        lambda: f"RENT PMT {random.choice(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])} {rand_num(4)}",
        lambda: f"LANDLORD TRANSFER {rand_num(6)}",
        lambda: f"PROPERTY MGMT {rand_num(4)}",
        lambda: f"HOME DEPOT #{rand_num(4)}",
        lambda: f"LOWES #{rand_num(4)}",
        lambda: f"IKEA {rand_num(6)}",
        lambda: f"WAYFAIR ORDER {rand_num(7)}",
        lambda: f"AIRBNB RESERVATION {rand_num(8)}",
        lambda: f"BOOKING.COM {rand_num(7)}",
        lambda: f"MORTGAGE PMT {rand_num(5)}",
        lambda: f"HOA FEE {rand_num(4)}",
        lambda: f"STORAGE UNIT {rand_num(4)}",
        lambda: f"ACE HARDWARE {rand_num(4)}",
        lambda: f"TRUE VALUE HW {rand_num(3)}",
        lambda: f"PLUMBER SVC {rand_num(4)}",
        lambda: f"PEST CONTROL {rand_num(4)}",
        lambda: f"MENARDS #{rand_num(4)}",
        lambda: f"SHERWIN WILLIAMS {rand_num(4)}",
        lambda: f"SECURITY DEPOSIT {rand_num(5)}",
        lambda: f"APARTMENT FEES {rand_num(4)}",
    ],
    "Entertainment": [
        lambda: f"NETFLIX.COM",
        lambda: f"SPOTIFY {rand_num(8)}",
        lambda: f"HULU {rand_num(7)}",
        lambda: f"DISNEY+ {rand_num(7)}",
        lambda: f"HBO MAX {rand_num(6)}",
        lambda: f"APPLE TV+ {rand_num(7)}",
        lambda: f"AMC THEATRES #{rand_num(4)}",
        lambda: f"REGAL CINEMA {rand_num(4)}",
        lambda: f"STEAM GAMES {rand_num(8)}",
        lambda: f"PLAYSTATION STORE",
        lambda: f"XBOX GAME PASS",
        lambda: f"NINTENDO ESHOP {rand_num(6)}",
        lambda: f"TWITCH.TV SUB {rand_num(5)}",
        lambda: f"TICKETMASTER {rand_num(8)}",
        lambda: f"STUBHUB #{rand_num(7)}",
        lambda: f"EVENTBRITE {rand_num(6)}",
        lambda: f"AUDIBLE CHARGE",
        lambda: f"KINDLE STORE {rand_num(6)}",
        lambda: f"ESPN+ SUBSCRIPTION",
        lambda: f"YOUTUBE PREMIUM",
    ],
    "Healthcare": [
        lambda: f"CVS PHARMACY #{rand_num(4)}",
        lambda: f"WALGREENS #{rand_num(4)}",
        lambda: f"RITE AID #{rand_num(4)}",
        lambda: f"DR {random.choice(['SMITH','JONES','PATEL','NGUYEN','GARCIA'])} MD {rand_num(4)}",
        lambda: f"CITY MEDICAL CTR {rand_num(4)}",
        lambda: f"URGENT CARE {rand_num(4)}",
        lambda: f"DENTAL OFFICE {rand_num(4)}",
        lambda: f"VISION CENTER #{rand_num(3)}",
        lambda: f"LABCORP {rand_num(7)}",
        lambda: f"QUEST DIAGNOSTICS {rand_num(5)}",
        lambda: f"HEALTH INS PMT {rand_num(5)}",
        lambda: f"CIGNA PREMIUM {rand_num(5)}",
        lambda: f"AETNA COPAY {rand_num(5)}",
        lambda: f"PHYSICAL THERAPY {rand_num(4)}",
        lambda: f"CHIROPRACTOR {rand_num(4)}",
        lambda: f"MENTAL HLTH SVCS {rand_num(4)}",
        lambda: f"MRI CENTER {rand_num(5)}",
        lambda: f"AMBULANCE SVC {rand_num(5)}",
        lambda: f"HOSPITAL BILL {rand_num(6)}",
        lambda: f"GNC SUPPLEMENT {rand_num(4)}",
    ],
    "Shopping": [
        lambda: f"AMAZON MKTPLACE {rand_num(8)}",
        lambda: f"AMAZON.COM*{rand_num(9)}",
        lambda: f"WALMART #{rand_num(4)}",
        lambda: f"TARGET #{rand_num(4)}",
        lambda: f"COSTCO WHSE #{rand_num(4)}",
        lambda: f"EBAY PURCHASE {rand_num(8)}",
        lambda: f"ETSY INC {rand_num(7)}",
        lambda: f"BEST BUY #{rand_num(4)}",
        lambda: f"APPLE.COM/BILL",
        lambda: f"MACYS #{rand_num(4)}",
        lambda: f"NORDSTROM #{rand_num(4)}",
        lambda: f"ZARA {rand_num(4)}",
        lambda: f"H&M #{rand_num(4)}",
        lambda: f"UNIQLO {rand_num(4)}",
        lambda: f"KOHLS #{rand_num(4)}",
        lambda: f"MARSHALLS #{rand_num(4)}",
        lambda: f"TJ MAXX #{rand_num(4)}",
        lambda: f"NIKE.COM {rand_num(8)}",
        lambda: f"ADIDAS #{rand_num(6)}",
        lambda: f"SHEIN ORDER {rand_num(7)}",
    ],
    "Education": [
        lambda: f"COURSERA SUB {rand_num(6)}",
        lambda: f"UDEMY COURSE {rand_num(7)}",
        lambda: f"LINKEDIN LRN {rand_num(6)}",
        lambda: f"SKILLSHARE PMT {rand_num(5)}",
        lambda: f"DUOLINGO PLUS",
        lambda: f"MASTERCLASS SUB",
        lambda: f"UNIVERSITY TUITION {rand_num(5)}",
        lambda: f"COMMUNITY COLLEGE {rand_num(4)}",
        lambda: f"STUDENT LOAN PMT {rand_num(6)}",
        lambda: f"CHEGG STUDY {rand_num(6)}",
        lambda: f"PEARSON EBOOK {rand_num(6)}",
        lambda: f"CAMPUS BOOKSTORE {rand_num(4)}",
        lambda: f"KHAN ACADEMY",
        lambda: f"EDX COURSE {rand_num(6)}",
        lambda: f"PLURALSIGHT {rand_num(7)}",
        lambda: f"CODECADEMY PRO",
        lambda: f"TUTORING SVC {rand_num(4)}",
        lambda: f"EXAM FEE {rand_num(5)}",
        lambda: f"SCHOOL SUPPLIES {rand_num(4)}",
        lambda: f"STUDY.COM {rand_num(6)}",
    ],
    "Utilities": [
        lambda: f"ELECTRIC BILL {rand_num(5)}",
        lambda: f"WATER UTILITY {rand_num(5)}",
        lambda: f"GAS UTILITY {rand_num(5)}",
        lambda: f"COMCAST #{rand_num(5)}",
        lambda: f"XFINITY INTERNET",
        lambda: f"AT&T BILL {rand_num(6)}",
        lambda: f"VERIZON WIRELESS",
        lambda: f"T-MOBILE {rand_num(6)}",
        lambda: f"SPECTRUM {rand_num(5)}",
        lambda: f"WASTE MGMT {rand_num(5)}",
        lambda: f"SEWAGE SVC {rand_num(4)}",
        lambda: f"INTERNET SVC {rand_num(5)}",
        lambda: f"CENTURYLINK {rand_num(6)}",
        lambda: f"COX COMM {rand_num(5)}",
        lambda: f"DISH NETWORK {rand_num(5)}",
        lambda: f"DIRECTV {rand_num(6)}",
        lambda: f"GOOGLE FI {rand_num(6)}",
        lambda: f"MINT MOBILE {rand_num(5)}",
        lambda: f"TRASH PICKUP {rand_num(4)}",
        lambda: f"SOLAR PANEL PMT {rand_num(5)}",
    ],
}

# ── Amount ranges per category ──────────────────────────────────────────────
AMOUNT_RANGES = {
    "Food":          (5,    80),
    "Transport":     (8,   150),
    "Housing":       (300, 2000),
    "Entertainment": (5,    50),
    "Healthcare":    (15,  500),
    "Shopping":      (10,  400),
    "Education":     (20,  800),
    "Utilities":     (30,  250),
}

# ── Ambiguous (noise) descriptions ─────────────────────────────────────────
AMBIGUOUS = [
    lambda: f"PAYMENT {rand_num(6)}",
    lambda: f"TRANSFER {rand_num(8)}",
    lambda: f"POS PURCHASE {rand_num(5)}",
    lambda: f"DEBIT CARD {rand_num(6)}",
    lambda: f"ONLINE PMT {rand_num(7)}",
    lambda: f"MISCELLANEOUS {rand_num(4)}",
    lambda: f"SQ *{rand_num(8)}",
    lambda: f"PP*{rand_num(9)}",
    lambda: f"CHECKCARD {rand_num(6)}",
    lambda: f"RECURRING CHG {rand_num(5)}",
]

# ── Generate rows ───────────────────────────────────────────────────────────
NUM_ROWS   = 2000
NOISE_PCT  = 0.05
categories = list(TEMPLATES.keys())

rows = []

# Guarantee at least 15 unique descriptions per category first
for cat in categories:
    templates = TEMPLATES[cat]
    for tmpl in templates:
        amount = round(random.uniform(*AMOUNT_RANGES[cat]), 2)
        rows.append({
            "id":          None,
            "date":        random_date(),
            "description": tmpl(),
            "amount":      amount,
            "category":    cat,
        })

# Fill remaining rows
remaining = NUM_ROWS - len(rows)
noise_count = int(NUM_ROWS * NOISE_PCT)
normal_count = remaining - noise_count

for _ in range(normal_count):
    cat = random.choice(categories)
    tmpl = random.choice(TEMPLATES[cat])
    amount = round(random.uniform(*AMOUNT_RANGES[cat]), 2)
    rows.append({
        "id":          None,
        "date":        random_date(),
        "description": tmpl(),
        "amount":      amount,
        "category":    cat,
    })

for _ in range(noise_count):
    cat = random.choice(categories)
    amount = round(random.uniform(5, 500), 2)
    rows.append({
        "id":          None,
        "date":        random_date(),
        "description": random.choice(AMBIGUOUS)(),
        "amount":      amount,
        "category":    "Ambiguous",
    })

random.shuffle(rows)

for i, row in enumerate(rows, start=1):
    row["id"] = i

# ── Write CSV ───────────────────────────────────────────────────────────────
OUTPUT = "transactions.csv"
fieldnames = ["id", "date", "description", "amount", "category"]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ── Summary ─────────────────────────────────────────────────────────────────
from collections import Counter
counts = Counter(r["category"] for r in rows)

print(f"\n{'='*42}")
print(f"  Synthetic Expense Dataset — Summary")
print(f"{'='*42}")
print(f"  Output file : {OUTPUT}")
print(f"  Total rows  : {len(rows)}")
print(f"{'─'*42}")
print(f"  {'Category':<16} {'Rows':>6}  {'Share':>6}")
print(f"{'─'*42}")
for cat in sorted(counts, key=lambda c: -counts[c]):
    pct = counts[cat] / len(rows) * 100
    print(f"  {cat:<16} {counts[cat]:>6}  {pct:>5.1f}%")
print(f"{'='*42}\n")
