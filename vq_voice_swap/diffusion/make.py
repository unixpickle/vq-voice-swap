from .schedule import CosSchedule, ExpSchedule, Schedule


def make_schedule(name: str) -> Schedule:
    """
    Create a schedule from a human-readable name.
    """
    if name == "exp":
        return ExpSchedule()
    elif name == "cos":
        return CosSchedule()
    else:
        raise ValueError(f"unknown schedule: {name}")
