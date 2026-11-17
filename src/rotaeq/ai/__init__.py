"""Reference implementations of the AI modules specified in Table A1 of the manuscript.

These modules define the configuration recommended for deployment and field
validation. In the manuscript, no AI model was trained on facility data; the effect
of each module was represented by MEASUR scenario parameters (Section 3.3.6).

The demo script (scripts/demo_ai_modules.py) exercises the pipelines on clearly
labelled synthetic data only. Replace it with field data (data/field/) to train
and validate the models.
"""
from .splits import chronological_split, group_split
from .features import vibration_features
from .models import (fit_mlp_with_early_stopping, make_demand_forecaster,
                     make_pump_health_classifier, make_torque_regressor)
from .setpoints import (fan_speed_setpoint, mixer_speed_setpoint,
                        pressure_setpoint_bar)

__all__ = [
    "chronological_split", "group_split", "vibration_features",
    "fit_mlp_with_early_stopping", "make_demand_forecaster",
    "make_pump_health_classifier", "make_torque_regressor",
    "fan_speed_setpoint", "mixer_speed_setpoint", "pressure_setpoint_bar",
]
