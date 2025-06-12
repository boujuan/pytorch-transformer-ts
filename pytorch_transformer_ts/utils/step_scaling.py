import logging

logger = logging.getLogger(__name__)

# Convert fractional warmup/decay steps to absolute values
def resolve_steps(value, total_steps, param_name):
    if value is None:
        return None
    elif isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
        # Treat as fraction of the stage (including 1.0 = 100%)
        absolute_value = int(value * total_steps)
        logger.info(f"  {param_name}: {value} (fraction) → {absolute_value:,} steps")
        return absolute_value
    elif isinstance(value, (int, float)) and value > 1.0:
        # Treat as absolute value (> 1.0)
        absolute_value = int(value)
        logger.info(f"  {param_name}: {absolute_value:,} steps (absolute)")
        return absolute_value
    else:
        logger.warning(f"  {param_name}: Invalid value {value}, using None")
        return None
