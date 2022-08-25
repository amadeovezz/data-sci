

def one_zero_loss(x: int) -> int:
    if x >= 0:
        return 0
    return 1


def hinge_loss(x: float) -> float:
    if x >= 1:
        return 0
    return 1 - x


def squared_error_loss(x: float) -> float:
    return x**2/2