"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Посчитайте корреляции Пирсона признаков с MRI_Count отдельно по полу."""
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    # Корреляции pandas рассчитываются попарно, поэтому строки с NA
    # исключаются только из конкретной пары признаков.
    women = df.loc[df["Gender"].astype(str).str.strip().str.lower() == "female"]
    men = df.loc[df["Gender"].astype(str).str.strip().str.lower() == "male"]

    def correlations(group: pd.DataFrame) -> dict[str, float]:
        values = group[FEATURES + ["MRI_Count"]].corr(method="pearson")["MRI_Count"]
        return {feature: float(values[feature]) for feature in FEATURES}

    women_correlations = correlations(women)
    men_correlations = correlations(men)

    strongest_mri_feature = max(
        FEATURES,
        key=lambda feature: max(
            abs(women_correlations[feature]),
            abs(men_correlations[feature]),
        ),
    )

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_correlations,
        men_mri_correlation=men_correlations,
        strongest_mri_feature=strongest_mri_feature,
    )
