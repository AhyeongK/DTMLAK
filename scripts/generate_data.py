import random
from pathlib import Path

import pandas as pd

# Reproducibility
random.seed(42)

NUMBER_OF_RECORDS = 5000
LABEL_NOISE_RATE = 0.06

# Manufacturing categories
processes = [
    "Formation",
    "Cell Loading",
    "Packaging",
    "Quality Inspection",
    "Equipment Setup",
]

shifts = [
    "Day",
    "Night",
]

levels = [
    "New",
    "Intermediate",
    "Experienced",
]

records = []

# Helper function for label noise
def apply_label_noise(priority: str) -> str:
    """
    Randomly changes a small percentage of labels.

    This represents unobserved manufacturing conditions,
    measurement uncertainty, and individual variation.
    """

    if random.random() >= LABEL_NOISE_RATE:
        return priority

    possible_labels = [
        label
        for label in ["Low", "Moderate", "High"]
        if label != priority
    ]

    # Prefer changing to a neighboring priority rather than
    # jumping directly from Low to High.
    if priority == "Low":
        weights = [0.90, 0.10]
    elif priority == "Moderate":
        weights = [0.50, 0.50]
    else:
        weights = [0.90, 0.10]

    return random.choices(
        possible_labels,
        weights=weights,
        k=1,
    )[0]


# Generate synthetic manufacturing records
for i in range(NUMBER_OF_RECORDS):

    operator = f"OP{i + 1:04d}"

    process = random.choice(processes)
    shift = random.choice(shifts)

    level = random.choices(
        levels,
        weights=[0.25, 0.35, 0.40],
        k=1,
    )[0]

    # Experience by operator level
    if level == "New":
        experience = random.randint(0, 8)

    elif level == "Intermediate":
        experience = random.randint(9, 24)

    else:
        experience = random.randint(25, 60)

    # Add occasional inconsistency between title and
    # experience to make the synthetic records less perfect.
    if random.random() < 0.05:
        experience += random.randint(-4, 4)
        experience = max(0, min(experience, 60))

    # Training score
    if level == "New":
        training_mean = 69

    elif level == "Intermediate":
        training_mean = 80

    else:
        training_mean = 88

    training = round(
        random.gauss(
            training_mean,
            9,
        )
    )

    if shift == "Night":
        training -= random.randint(0, 4)

    # Measurement or assessment variation
    training += round(
        random.gauss(
            0,
            3,
        )
    )

    training = max(
        40,
        min(training, 100),
    )

    # Production volume
    units = random.randint(
        80,
        150,
    )

    # Base defect tendency by process
    process_defect_means = {
        "Equipment Setup": 3.2,
        "Formation": 2.8,
        "Cell Loading": 2.4,
        "Quality Inspection": 2.1,
        "Packaging": 1.8,
    }

    defect_mean = process_defect_means[process]

    # New and night-shift records receive only a small
    # tendency adjustment, not a guaranteed outcome.
    if level == "New":
        defect_mean += 0.8

    elif level == "Experienced":
        defect_mean -= 0.4

    if shift == "Night":
        defect_mean += 0.4

    if training < 65:
        defect_mean += 0.8

    elif training >= 90:
        defect_mean -= 0.3

    defects = max(
        0,
        round(
            random.gauss(
                defect_mean,
                1.8,
            )
        ),
    )

    defects = min(
        defects,
        units,
    )

    # Rework tendency
    rework_mean = defects * random.uniform(
        0.35,
        0.70,
    )

    if process == "Equipment Setup":
        rework_mean += 0.4

    if shift == "Night":
        rework_mean += 0.2

    rework = max(
        0,
        round(
            random.gauss(
                rework_mean,
                1.2,
            )
        ),
    )

    rework = min(
        rework,
        units,
    )

    # Latent support score
    support_score = 0.0

    # Smooth effects instead of only fixed cutoffs
    support_score += max(
        0,
        75 - training,
    ) * 0.10

    support_score += max(
        0,
        12 - experience,
    ) * 0.12

    support_score += defects * 0.42
    support_score += rework * 0.58

    if shift == "Night":
        support_score += 0.45

    if level == "New":
        support_score += 0.65

    elif level == "Experienced":
        support_score -= 0.25

    if process == "Equipment Setup":
        support_score += 0.50

    elif process == "Formation":
        support_score += 0.25

    # Unobserved manufacturing variation
    support_score += random.gauss(
        0,
        0.85,
    )

    # Convert score to support priority
    if support_score >= 5.4:
        priority = "High"

    elif support_score >= 2.8:
        priority = "Moderate"

    else:
        priority = "Low"

    # Add 6% label uncertainty
    priority = apply_label_noise(priority)

    records.append(
        {
            "Operator_ID": operator,
            "Process": process,
            "Shift": shift,
            "Operator_Level": level,
            "Training_Score": training,
            "Experience_Months": experience,
            "Total_Units": units,
            "Defect_Count": defects,
            "Rework_Count": rework,
            "Support_Priority": priority,
        }
    )


# Create and save dataset
df = pd.DataFrame(records)

output_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic_manufacturing_data.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    output_path,
    index=False,
)

# Display summary
print("=" * 70)
print("SYNTHETIC MANUFACTURING DATASET GENERATED")
print("=" * 70)

print(f"\nRecords generated: {len(df):,}")

print("\nSample records:")
print(df.head())

print("\nSupport priority distribution:")
print(df["Support_Priority"].value_counts())

print("\nSupport priority percentage:")
print(
    (
        df["Support_Priority"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
    )
)

print(
    f"\nConfigured label noise rate: "
    f"{LABEL_NOISE_RATE:.0%}"
)

print(f"\nSaved to:\n{output_path}")