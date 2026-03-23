"""
MoMA Preprocessing: Donor x Medium Clusters + Expected Display Model
SI 649, University of Michigan
Usage:  python preprocess.py --artworks Artworks.csv
"""

import pandas as pd
import numpy as np
import re
import json
import argparse
import os
import sys
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFIED BOARD LIST (moma.org/about/trustees + Annual Reports 2015-2024)
# ═══════════════════════════════════════════════════════════════════════════════

BOARD_CURRENT = [
    "Marie-Josée Kravis", "Sarah Arison", "Sid R. Bass", "Mimi Haas",
    "Marlene Hess", "Maja Oeri", "Edgar Wachenheim III",
    "Ronald S. Lauder", "Jerry I. Speyer", "Agnes Gund",
    "Alexandre Arnault", "Scott Belsky", "Lawrence B. Benenson",
    "Leon D. Black", "David Booth", "Clarissa Alcock Bronfman",
    "Patricia Phelps de Cisneros", "Steven Cohen", "Edith Cooper",
    "Paula Crown", "David Dechman", "Anne Dias Griffin", "Glenn Dubin",
    "Lonti Ebers", "Joel S. Ehrenkranz", "John Elkann", "Laurence Fink",
    "Glenn Fuhrman", "Kathleen Fuld", "David Grain",
    "Ronnie Heyman", "AC Hudgins", "Pamela Joyner", "Jill Kraus",
    "Micky Malka", "Khalil Gibran Muhammad", "Philip S. Niarchos",
    "James G. Niven", "Peter Norton", "Daniel S. Och",
    "Eyal Ofer", "Michael S. Ovitz", "Emily Rauh Pulitzer",
    "Sharon Percy Rockefeller", "Richard Roth", "Richard E. Salomon",
    "Anna Marie Shapiro", "Anna Deavere Smith", "Robert Soros",
    "Jon Stryker", "Daniel Sundheim", "Tony Tamer", "Steven Tananbaum",
    "Alice M. Tisch",
    "Lin Arison", "Elizabeth Diller", "Maurice R. Greenberg",
    "Wynton Marsalis", "Ted Sann", "Yoshio Taniguchi",
    "Wallis Annenberg",
]

BOARD_PAST = [
    "Eli Broad", "Douglas S. Cramer", "Lewis B. Cullman",
    "Gianluigi Gabetti", "Barbara Jakobson", "Werner H. Kramarsky",
    "June Noble Larkin", "Thomas H. Lee", "Donald B. Marron",
    "Robert B. Menschel", "Peter G. Peterson", "David Rockefeller",
    "David Rockefeller Jr.", "Jeanne C. Thayer", "Joan Tisch",
    "Gilbert Silverman", "Lord Rogers of Riverside", "Eugene V. Thaw",
    "Celeste Bartos", "Gary Winnick", "Xin Zhang",
    "William S. Paley", "Philip Johnson", "Nelson A. Rockefeller",
    "Blanchette Hooker Rockefeller",
]

BOARD_FOUNDING = [
    "Abby Aldrich Rockefeller", "Lillie P. Bliss", "Mary Quinn Sullivan",
    "A. Conger Goodyear", "Mrs. Simon Guggenheim",
    "Mrs. John D. Rockefeller Jr.", "Mrs. John D. Rockefeller 3rd",
    "John Hay Whitney",
]

ALL_BOARD = BOARD_CURRENT + BOARD_PAST + BOARD_FOUNDING

def normalize(name):
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return " ".join(name.split())

BOARD_NORM_SET = set()
BOARD_NORM_LIST = []
BOARD_LAST_TO_FULL = {}
for bm in ALL_BOARD:
    n = normalize(bm)
    BOARD_NORM_SET.add(n)
    BOARD_NORM_LIST.append(n)
    parts = n.split()
    if parts:
        BOARD_LAST_TO_FULL.setdefault(parts[-1], []).append(n)

DISTINCTIVE_LAST = {
    "rockefeller", "guggenheim", "lauder", "gund", "paley", "kravis",
    "broad", "bliss", "sullivan", "niarchos", "ovitz", "cisneros",
    "pulitzer", "bartos", "menschel", "marron", "bronfman", "oeri",
    "salomon", "heyman", "hess", "taniguchi", "elkann", "dubin",
    "fuhrman", "greenberg", "jakobson", "annenberg", "speyer",
    "ehrenkranz", "wachenheim", "stryker", "sundheim", "tananbaum",
    "arnault", "crowninshield", "goodyear",
}

def is_board_connected(donor_name):
    norm = normalize(donor_name)
    if norm in BOARD_NORM_SET:
        return True
    for bm in BOARD_NORM_LIST:
        if bm in norm or norm in bm:
            return True
    donor_words = set(norm.split())
    for last, fulls in BOARD_LAST_TO_FULL.items():
        if last in donor_words and len(last) > 3:
            if last in DISTINCTIVE_LAST:
                return True
            for full in fulls:
                ff = full.split()[0]
                for w in donor_words:
                    if w and w[0] == ff[0] and w != last:
                        return True
    for pat in [r"mrs\.?\s+john\s+d\.?\s+rockefeller",
                r"mrs\.?\s+simon\s+guggenheim",
                r"mrs\.?\s+david\s+rockefeller"]:
        if re.search(pat, donor_name.lower()):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# DONOR EXTRACTION (with filtering for generic / non-person names)
# ═══════════════════════════════════════════════════════════════════════════════

# These are NOT real donor names. They are generic descriptions that the
# regex picks up from CreditLine strings.
GENERIC_DONORS = {
    "the artist", "the designer", "the architect", "the manufacturer",
    "the publisher", "the producer", "the photographer", "the printer",
    "the editor", "the author", "the composer", "the director",
    "the estate", "the family", "the foundation",
    "various donors", "anonymous", "unknown",
    "the filmmaker", "the choreographer",
}

def is_generic_donor(name):
    """Return True if this is a generic role description, not a real person."""
    low = name.lower().strip()
    # Direct match
    if low in GENERIC_DONORS:
        return True
    # Starts with "the " and is short / generic-sounding
    if low.startswith("the ") and len(low.split()) <= 3:
        # Check if the rest is a generic role word
        rest = low[4:].strip()
        generic_words = {
            "artist", "designer", "architect", "manufacturer", "publisher",
            "producer", "photographer", "printer", "editor", "author",
            "composer", "director", "filmmaker", "choreographer",
            "estate", "family", "foundation", "company", "firm",
            "artists", "designers", "architects", "manufacturers",
        }
        if rest in generic_words:
            return True
    # "Family of Man", "Family of [something]" is a photo exhibition title
    if re.match(r"(?:the\s+)?family\s+of\s+", low):
        return True
    # Very short names (1-2 chars) or just initials
    if len(low.replace(" ", "")) < 3:
        return True
    return False


def extract_donors(credit_line):
    if pd.isna(credit_line):
        return []
    cl = str(credit_line).strip()
    donors = []
    for pat in [
        r"(?:Gift|gift|Bequest|bequest|Promised gift|promised gift)\s+of\s+(.+?)(?:\.|,\s*\d|\sin\s+(?:honor|memory|exchange)|$)",
    ]:
        for m in re.findall(pat, cl):
            name = m.strip().rstrip(".,;:")
            if not is_generic_donor(name):
                donors.append(name)
    for m in re.findall(r"[Tt]he\s+(.+?)\s+Fund", cl):
        name = m.strip()
        if not is_generic_donor(name):
            donors.append(name)
    if not donors and re.search(r"\bPurchase\b", cl, re.IGNORECASE):
        donors.append("__PURCHASE__")
    return donors


def simplify_classification(cls):
    if pd.isna(cls): return "Other"
    cls = str(cls).strip()
    return {"Mies van der Rohe Archive": "Architecture",
            "Frank Lloyd Wright Archive": "Architecture",
            "A&D Cataloged": "Architecture",
            "(not assigned)": "Other",
            "Multiple Coversheet": "Other"}.get(cls, cls)

MEDIUM_ORDER = ["Painting", "Sculpture", "Drawing", "Design",
                "Architecture", "Print", "Photograph"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artworks", default="Artworks.csv")
    parser.add_argument("--output", default="moma_donor_viz.json")
    args = parser.parse_args()

    if not os.path.exists(args.artworks):
        print(f"ERROR: '{args.artworks}' not found.")
        print("Download from https://github.com/MuseumofModernArt/collection")
        sys.exit(1)

    print(f"Loading {args.artworks}...")
    df = pd.read_csv(args.artworks, low_memory=False)
    print(f"  {len(df)} artworks loaded.")

    df["Classification"] = df["Classification"].apply(simplify_classification)
    df["OnView"] = df["OnView"].notna()
    df["YearAcquired"] = pd.to_datetime(df["DateAcquired"], errors="coerce").dt.year

    def extract_gender(g):
        if pd.isna(g): return "Unknown"
        s = str(g).lower()
        if "female" in s: return "Female"
        if "male" in s: return "Male"
        return "Unknown"
    def extract_nationality(nat):
        if pd.isna(nat): return "Unknown"
        m = re.search(r"\(([^)]+)\)", str(nat))
        return m.group(1) if m else "Unknown"

    df["ArtistGender"] = df["Gender"].apply(extract_gender)
    df["ArtistNationality"] = df["Nationality"].apply(extract_nationality)

    # ── Per-factor display rates ─────────────────────────────────────────
    print("Computing factor display rates...")
    medium_rates = df.groupby("Classification")["OnView"].mean().to_dict()
    gender_rates = df.groupby("ArtistGender")["OnView"].mean().to_dict()

    nat_counts = df["ArtistNationality"].value_counts()
    common_nats = set(nat_counts[nat_counts >= 100].index)
    df["NatGroup"] = df["ArtistNationality"].apply(lambda x: x if x in common_nats else "__Other__")
    nat_rates = df.groupby("NatGroup")["OnView"].mean().to_dict()

    df["DecadeAcq"] = (df["YearAcquired"] // 10 * 10).fillna(0).astype(int)
    decade_rates = df.groupby("DecadeAcq")["OnView"].mean().to_dict()

    overall_rate = df["OnView"].mean()

    def expected_rate(row):
        mr = medium_rates.get(row["Classification"], overall_rate)
        gr = gender_rates.get(row["ArtistGender"], overall_rate)
        nr = nat_rates.get(row["NatGroup"], overall_rate)
        dr = decade_rates.get(row["DecadeAcq"], overall_rate)
        return 0.50 * mr + 0.15 * gr + 0.15 * nr + 0.20 * dr

    df["ExpectedRate"] = df.apply(expected_rate, axis=1)

    WESTERN = {"American", "Unknown", "British", "French", "German", "Italian",
               "Austrian", "Swiss", "Dutch", "Belgian", "Canadian", "Australian",
               "Swedish", "Danish", "Norwegian", "Finnish", "Irish", "Scottish"}
    df["IsNonWestern"] = ~df["ArtistNationality"].isin(WESTERN)
    df["IsFemale"] = df["ArtistGender"] == "Female"
    df["IsRecent"] = df["YearAcquired"] >= 2000

    # ── Extract donors and explode ───────────────────────────────────────
    print("Extracting donors...")
    df["DonorList"] = df["CreditLine"].apply(extract_donors)

    rows = []
    for _, row in df.iterrows():
        donors = row["DonorList"] if row["DonorList"] else ["__NONE__"]
        for d in donors:
            rows.append({
                "ObjectID": row["ObjectID"],
                "Classification": row["Classification"],
                "OnView": row["OnView"],
                "ArtistGender": row["ArtistGender"],
                "ArtistNationality": row["ArtistNationality"],
                "Donor": d,
                "ExpectedRate": row["ExpectedRate"],
                "IsNonWestern": row["IsNonWestern"],
                "IsFemale": row["IsFemale"],
                "IsRecent": row["IsRecent"],
                "YearAcquired": row["YearAcquired"],
            })

    dfe = pd.DataFrame(rows)
    real = dfe[~dfe["Donor"].isin(["__NONE__", "__PURCHASE__"])].copy()
    real["IsBoard"] = real["Donor"].apply(is_board_connected)

    board_donors_found = sorted(real[real["IsBoard"]]["Donor"].unique().tolist())
    print(f"  {real['Donor'].nunique()} donors. {len(board_donors_found)} board-matched.")
    print(f"  Sample board donors: {board_donors_found[:10]}")

    # ── Aggregate per donor x medium ─────────────────────────────────────
    real_med = real[real["Classification"].isin(MEDIUM_ORDER)]

    clusters = real_med.groupby(["Donor", "Classification"]).agg(
        totalWorks=("ObjectID", "nunique"),
        worksOnView=("OnView", "sum"),
        isBoard=("IsBoard", "first"),
        pctFemale=("ArtistGender", lambda x: round((x == "Female").mean(), 3)),
        pctNonWestern=("IsNonWestern", "mean"),
        pctRecent=("IsRecent", "mean"),
        topNationality=("ArtistNationality", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
        expectedRate=("ExpectedRate", "mean"),
        medianYear=("YearAcquired", "median"),
    ).reset_index()

    clusters["worksOnView"] = clusters["worksOnView"].astype(int)
    clusters["displayRate"] = (clusters["worksOnView"] / clusters["totalWorks"]).round(4)
    clusters["expectedRate"] = clusters["expectedRate"].round(4)
    clusters["residual"] = (clusters["displayRate"] - clusters["expectedRate"]).round(4)
    clusters["pctNonWestern"] = clusters["pctNonWestern"].round(3)
    clusters["pctRecent"] = clusters["pctRecent"].round(3)
    clusters["medianYear"] = clusters["medianYear"].round(0)
    clusters.rename(columns={"Classification": "medium", "Donor": "name"}, inplace=True)

    # Reason tags
    def make_reason(row):
        tags = []
        if row["isBoard"]:
            tags.append("board")
        if row["medium"] in ("Painting", "Sculpture"):
            tags.append("medium")
        if row["pctNonWestern"] > 0.4:
            tags.append("diversity")
        if row["pctFemale"] > 0.35:
            tags.append("gender")
        if row["pctRecent"] > 0.5:
            tags.append("recent")
        return tags

    clusters["reasonTags"] = clusters.apply(make_reason, axis=1)

    # Dominant non-board reason
    def dominant_content_reason(row):
        if row["isBoard"]:
            return "board"
        if row["pctNonWestern"] > 0.4:
            return "diversity"
        if row["pctRecent"] > 0.5:
            return "recent"
        if row["pctFemale"] > 0.35:
            return "gender"
        if row["medium"] in ("Painting", "Sculpture"):
            return "medium"
        return "none"

    clusters["dominantReason"] = clusters.apply(dominant_content_reason, axis=1)

    # Filter: >= 5 works AND (on view > 0 OR board with >= 5 works)
    viz = clusters[
        (clusters["totalWorks"] >= 5) &
        ((clusters["worksOnView"] > 0) | (clusters["isBoard"]))
    ].copy()

    # ── Medium stats (for proportional row heights) ──────────────────────
    med_stats = df[df["Classification"].isin(MEDIUM_ORDER)].groupby("Classification").agg(
        total=("ObjectID", "count"), onView=("OnView", "sum"),
    ).reset_index()
    med_stats["displayRate"] = (med_stats["onView"] / med_stats["total"]).round(4)
    med_stats["pctCollection"] = (med_stats["total"] / len(df)).round(4)
    med_stats.rename(columns={"Classification": "classification"}, inplace=True)
    med_stats["onView"] = med_stats["onView"].astype(int)

    # ── Summaries ────────────────────────────────────────────────────────
    def summary(subset):
        n = int(subset["ObjectID"].nunique())
        v = int(subset[subset["OnView"]]["ObjectID"].nunique())
        return {"total_works": n, "on_view": v, "display_rate": round(v / max(n, 1), 4)}

    output = {
        "clusters": json.loads(viz.to_json(orient="records")),
        "mediumOrder": MEDIUM_ORDER,
        "mediums": json.loads(med_stats.to_json(orient="records")),
        "boardSummary": summary(real[real["IsBoard"]]),
        "otherSummary": summary(real[~real["IsBoard"]]),
        "totalArtworks": int(len(df)),
        "totalOnView": int(df["OnView"].sum()),
        "overallDisplayRate": round(float(df["OnView"].mean()), 4),
        "boardDonorNames": board_donors_found,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {args.output}")
    print(f"  {len(viz)} clusters (>= 5 works)")
    # Quick check for bogus donors
    print(f"\n  Sample non-board clusters with display:")
    sample = viz[(~viz["isBoard"]) & (viz["worksOnView"] > 0)].nlargest(8, "worksOnView")
    for _, r in sample.iterrows():
        print(f"    {r['name']} | {r['medium']} | {r['totalWorks']} donated, {r['worksOnView']} on view | reason: {r['dominantReason']}")


if __name__ == "__main__":
    main()
