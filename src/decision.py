"""Decision layer: map RUL + risk to recommended actions using a cost model."""
from __future__ import annotations

from typing import Dict

from src.config import (
    COST_EARLY_MAINTENANCE,
    COST_INSPECTION,
    COST_UNPLANNED_FAILURE,
    THRESH_INSPECT,
    THRESH_OK,
)

"""
Cost assumptions (documented):
- Unplanned failure (engine fails in service): COST_UNPLANNED_FAILURE = 100 units
- Performing maintenance earlier than necessary: COST_EARLY_MAINTENANCE = 10 units
- Inspection / scheduled check: COST_INSPECTION = 2 units

Thresholds chosen so that expected cost of waiting exceeds inspection cost
when predicted RUL is low. Simple rule-based policy (not optimized):
  RUL > THRESH_OK (60)      → OK
  THRESH_INSPECT < RUL ≤ 60 → Schedule Inspection
  RUL ≤ THRESH_INSPECT (30) → Urgent maintenance
"""


def recommend_action(
    rul_pred: float,
    fails_soon_prob: float = 0.0,
    health_index: float = 0.0,
) -> Dict[str, str]:
    """
    Return action label and short rationale.
    """
    if rul_pred <= THRESH_INSPECT or fails_soon_prob > 0.7 or health_index > 0.8:
        action = "Urgent"
        reason = (
            f"Predicted RUL={rul_pred:.0f} ≤ {THRESH_INSPECT} "
            f"or high failure risk / health index."
        )
    elif rul_pred <= THRESH_OK or fails_soon_prob > 0.4 or health_index > 0.5:
        action = "Schedule Inspection"
        reason = (
            f"Predicted RUL={rul_pred:.0f} in ({THRESH_INSPECT}, {THRESH_OK}]; "
            "inspection cost is low relative to failure cost."
        )
    else:
        action = "OK"
        reason = f"Predicted RUL={rul_pred:.0f} > {THRESH_OK}; continue operation."

    return {
        "action": action,
        "reason": reason,
        "cost_assumptions": (
            f"unplanned_failure={COST_UNPLANNED_FAILURE}, "
            f"early_maintenance={COST_EARLY_MAINTENANCE}, "
            f"inspection={COST_INSPECTION}"
        ),
    }
