

def one_zero(x: int) -> int:
    if x >= 0:
        return 0
    return 1


def hinge(x: float) -> float:
    if x >= 1:
        return 0
    return 1 - x


def squared_error(x: float) -> float:
    return x**2