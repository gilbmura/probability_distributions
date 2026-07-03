# ============================================================
# PART 2: BAYESIAN PROBABILITY - IMDb SENTIMENT ANALYSIS
# ============================================================
# Course: Formative 3 - Probability Distributions, Bayesian
#         Probability, and Gradient Descent Implementation
# Author: Isimbi Selena
# Dataset: IMDb Movie Reviews (50,000 reviews)
#
# GOAL
# ----
# For a small set of hand-picked keywords, estimate how much the
# presence of that keyword in a review shifts our belief that the
# review is POSITIVE, using Bayes' Theorem:
#
#           P(keyword | Positive) * P(Positive)
#   P(Positive | keyword) = -------------------------------------
#                              P(keyword)
#
# Only P(Positive | keyword) is computed, as required by the
# assignment brief (P(Negative | keyword) is intentionally
# NOT calculated).
#
# CONSTRAINTS
# -----------
# - Only Python's standard library is used (csv, re, collections).
#   No pandas, numpy, sklearn, etc.
# - Every repeated operation lives in exactly one function (DRY).
# ============================================================

import csv
import re
from collections import namedtuple

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = "IMDB Dataset.csv"

# Keyword selection with rationale:
#
# POSITIVE keywords - words a reviewer only reaches for when they
#   genuinely liked the film:
#   - "masterpiece" : the strongest possible praise a critic can give
#   - "hilarious"   : strong, unambiguous positive reaction (comedy)
#   - "brilliant"   : general-purpose praise for direction/acting/writing
#
# NEGATIVE keywords - words that signal disappointment or dislike:
#   - "boring"        : the single most common complaint in negative reviews
#   - "disappointing" : explicit statement that expectations were unmet
#   - "waste"         : shorthand for "waste of time/money", a strong pan
#
# CONTROL keyword - included to test the model on an ambiguous term:
#   - "funny" : can describe a comedy working exactly as intended
#               (positive) OR a film being unintentionally / badly funny
#               (negative), so its posterior is expected to sit closer
#               to the 0.50 prior than the other keywords.
POSITIVE_KEYWORDS = ["masterpiece", "hilarious", "brilliant"]
NEGATIVE_KEYWORDS = ["boring", "disappointing", "waste"]
CONTROL_KEYWORDS = ["funny"]
ALL_KEYWORDS = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS + CONTROL_KEYWORDS

POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"

# Container for one keyword's full set of results (avoids repeating
# the same dict-of-values pattern everywhere -> DRY).
BayesResult = namedtuple(
    "BayesResult",
    ["keyword", "category", "prior", "likelihood", "marginal", "posterior",
     "pos_count", "neg_count", "total_count"],
)


# ------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------

def load_imdb_reviews(filename=DATASET_PATH):
    """
    Load IMDb reviews from a CSV file with columns: review, sentiment.

    Returns a list of dicts: {'text': str, 'sentiment': str}
    Both fields are lower-cased and stripped for consistent matching.
    """
    reviews = []
    try:
        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # skip header row
            for row in reader:
                if len(row) >= 2:
                    reviews.append({
                        "text": row[0].strip().lower(),
                        "sentiment": row[1].strip().lower(),
                    })
        print(f"Successfully loaded {len(reviews):,} reviews from '{filename}'")
        return reviews
    except FileNotFoundError:
        print(f"ERROR: '{filename}' not found. Place the IMDb CSV next to this script.")
        return []
    except Exception as exc:
        print(f"ERROR while loading data: {exc}")
        return []


# ------------------------------------------------------------
# 2. REUSABLE HELPERS (DRY core: every count/probability funnels
#    through these few functions)
# ------------------------------------------------------------

def filter_by_sentiment(reviews, sentiment):
    """Return only the reviews matching a given sentiment label."""
    return [r for r in reviews if r["sentiment"] == sentiment]


def count_keyword_occurrences(reviews, keyword):
    """
    Count whole-word, case-insensitive occurrences of `keyword` across
    a list of reviews. Word boundaries (\\b) prevent partial matches
    (e.g. 'funny' should not match inside 'funnyman').
    """
    pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
    return sum(len(pattern.findall(r["text"])) for r in reviews)


def total_word_count(reviews):
    """Total number of alphabetic words across a list of reviews."""
    total = 0
    for r in reviews:
        total += len(re.findall(r"\b[a-z]+\b", r["text"]))
    return total


# ------------------------------------------------------------
# 3. BAYES' THEOREM
# ------------------------------------------------------------

def calculate_bayes_for_keyword(keyword, category, all_reviews,
                                 positive_reviews, negative_reviews,
                                 total_positive_words):
    """
    Compute Prior, Likelihood, Marginal, and Posterior for ONE keyword.
    This is the single place Bayes' Theorem is implemented; every
    keyword in ALL_KEYWORDS is passed through this same function (DRY).

        Prior      P(Positive)          = positive reviews / all reviews
        Likelihood P(keyword|Positive)   = keyword count in positive reviews
                                            / total words in positive reviews
        Marginal   P(keyword)            = keyword count in all reviews
                                            / total reviews
        Posterior  P(Positive|keyword)   = Likelihood * Prior / Marginal
    """
    total_reviews = len(all_reviews)

    # Prior: P(Positive)
    prior = len(positive_reviews) / total_reviews

    # Raw counts, needed both for likelihood/marginal and for the
    # "detailed counts" table later.
    pos_count = count_keyword_occurrences(positive_reviews, keyword)
    neg_count = count_keyword_occurrences(negative_reviews, keyword)
    total_count = pos_count + neg_count

    # Likelihood: P(keyword | Positive)
    likelihood = pos_count / total_positive_words if total_positive_words else 0.0

    # Marginal: P(keyword)  -- occurrences per review, across the whole corpus
    marginal = total_count / total_reviews

    # Posterior: P(Positive | keyword)  via Bayes' Theorem
    posterior = (likelihood * prior) / marginal if marginal else 0.0

    return BayesResult(
        keyword=keyword, category=category, prior=prior, likelihood=likelihood,
        marginal=marginal, posterior=posterior, pos_count=pos_count,
        neg_count=neg_count, total_count=total_count,
    )


def keyword_category(keyword):
    """Look up whether a keyword is POSITIVE / NEGATIVE / CONTROL."""
    if keyword in POSITIVE_KEYWORDS:
        return "POSITIVE"
    if keyword in NEGATIVE_KEYWORDS:
        return "NEGATIVE"
    return "CONTROL"


def run_bayesian_analysis(reviews):
    """
    Run the full Bayesian analysis for every keyword in ALL_KEYWORDS.
    Returns a dict {keyword: BayesResult}.
    """
    positive_reviews = filter_by_sentiment(reviews, POSITIVE_LABEL)
    negative_reviews = filter_by_sentiment(reviews, NEGATIVE_LABEL)
    total_positive_words = total_word_count(positive_reviews)

    results = {}
    for keyword in ALL_KEYWORDS:
        results[keyword] = calculate_bayes_for_keyword(
            keyword, keyword_category(keyword), reviews,
            positive_reviews, negative_reviews, total_positive_words,
        )
    return results


# ------------------------------------------------------------
# 4. DISPLAY / REPORTING (all printing logic lives here, reused for
#    every table so formatting never gets duplicated)
# ------------------------------------------------------------

def print_probability_table(results):
    header = f"{'Keyword':<16}{'Category':<10}{'P(Positive)':<13}{'P(kw|Pos)':<14}{'P(kw)':<12}{'P(Pos|kw)':<12}"
    print("\n" + "=" * len(header))
    print("BAYESIAN PROBABILITY RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for keyword in ALL_KEYWORDS:
        r = results[keyword]
        print(f"{r.keyword:<16}{r.category:<10}{r.prior:<13.4f}"
              f"{r.likelihood:<14.6f}{r.marginal:<12.4f}{r.posterior:<12.3f}")
    print("=" * len(header))


def print_counts_table(results):
    header = f"{'Keyword':<16}{'Positive count':<16}{'Negative count':<16}{'Total count':<12}"
    print("\nDETAILED OCCURRENCE COUNTS")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for keyword in ALL_KEYWORDS:
        r = results[keyword]
        print(f"{r.keyword:<16}{r.pos_count:<16,}{r.neg_count:<16,}{r.total_count:<12,}")


def print_interpretation(results):
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    for label, group in (("POSITIVE", POSITIVE_KEYWORDS),
                          ("NEGATIVE", NEGATIVE_KEYWORDS),
                          ("CONTROL", CONTROL_KEYWORDS)):
        print(f"\n{label} keywords:")
        for kw in group:
            r = results[kw]
            share = (r.pos_count / r.total_count * 100) if r.total_count else 0
            print(f"  '{kw}': posterior P(Positive|{kw}) = {r.posterior:.3f} "
                  f"-> {share:.1f}% of its occurrences are in positive reviews")


def save_results_to_csv(results, filename="bayesian_results.csv"):
    """Persist the results table to CSV for easy sharing / grading."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Keyword", "Category", "Prior_P(Positive)",
                          "Likelihood_P(kw|Positive)", "Marginal_P(kw)",
                          "Posterior_P(Positive|kw)", "Positive_Count",
                          "Negative_Count", "Total_Count"])
        for keyword in ALL_KEYWORDS:
            r = results[keyword]
            writer.writerow([r.keyword, r.category, f"{r.prior:.4f}",
                              f"{r.likelihood:.8f}", f"{r.marginal:.4f}",
                              f"{r.posterior:.4f}", r.pos_count, r.neg_count,
                              r.total_count])
    print(f"\nResults saved to '{filename}'")


def print_markdown_table(results):
    """Print a GitHub-flavoured markdown version of the results table."""
    lines = [
        "| Keyword | Category | Prior P(Positive) | Likelihood P(kw\\|Positive) | Marginal P(kw) | Posterior P(Positive\\|kw) |",
        "|---------|----------|--------------------|------------------------------|-----------------|-----------------------------|",
    ]
    for keyword in ALL_KEYWORDS:
        r = results[keyword]
        lines.append(f"| {r.keyword} | {r.category} | {r.prior:.4f} | "
                      f"{r.likelihood:.6f} | {r.marginal:.4f} | {r.posterior:.3f} |")
    print("\n".join(lines))


# ------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------

def main():
    reviews = load_imdb_reviews()
    if not reviews:
        print("No data loaded - place 'IMDB Dataset.csv' next to this script and re-run.")
        return

    total = len(reviews)
    positive = len(filter_by_sentiment(reviews, POSITIVE_LABEL))
    negative = len(filter_by_sentiment(reviews, NEGATIVE_LABEL))
    print(f"\nDataset: {total:,} reviews  "
          f"({positive:,} positive / {negative:,} negative)")
    print(f"Positive keywords: {', '.join(POSITIVE_KEYWORDS)}")
    print(f"Negative keywords: {', '.join(NEGATIVE_KEYWORDS)}")
    print(f"Control keyword:   {', '.join(CONTROL_KEYWORDS)}")

    results = run_bayesian_analysis(reviews)

    print_probability_table(results)
    print_counts_table(results)
    print_interpretation(results)

    print("\nMarkdown table:")
    print_markdown_table(results)

    save_results_to_csv(results)


if __name__ == "__main__":
    main()
