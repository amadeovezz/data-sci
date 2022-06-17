

def one_zero_loss(x: int) -> int:
    if x >= 0:
        return 0
    return 1


def hinge_loss(x: int) -> int:
    if x >= 1:
        return 0
    return 1 - x

