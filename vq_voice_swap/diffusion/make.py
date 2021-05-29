from .schedule import Schedule, ExpSchedule


def make_schedule(name: str) -> Schedule:
    """
    Create a schedule from a human-readable name.
    """
    if name == "exp":
        return ExpSchedule()
    else:
        raise ValueError(f"unknown schedule: {name}")
