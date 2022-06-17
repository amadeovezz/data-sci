import math


def eulers_number(n):
    """
    Shows how approximations of eulers number gets better as the lim n -> infinity

    :param n: Natural number
    :return: None
    """

    # avoid scientific notation by casting to float
    for n in range(1, n):
        growth = (1 / float(n))
        e = (1 + (1 / float(n))) ** n
        print(f"when compounding = {n}, we have a growth rate = {growth}, e is: {e}")


def epsilon_delta_sqrt(c: float, limit: float, epsilon: float) -> bool:
    """
    Goal: Determines whether the limit exists as x -> c for a given epsilon.

    Description: Numeric epsilon delta solver for f(x) = sqrt(x).

    Uses small delta values to determine truth.
    """

    print(f"Verifying lim x->{c} sqrt(x) = {limit}\n")
    print(f"let epsilon be: {e}")

    # Find our epsilon boundary, this can be any epsilon
    # | f(x) - L | < e = L - e < f(x) < L + e

    print(f"Performing scratch work, looking for corresponding x interval for epsilon...")

    epsilon_l = limit + epsilon
    epsilon_r = limit - epsilon
    print(f"(L - epsilon, epsilon + L) interval: {epsilon_r} < {limit} < {epsilon_l}")

    # Find the inverse of f(x) = sqrt(x) , which is just f(x) = x^2
    # Get the corresponding x epsilon interval
    x_epsilon_l = epsilon_l ** 2
    x_epsilon_r = epsilon_r ** 2

    # Quick sanity check to make sure c in fact falls in-between our x epsilon interval
    try:
        assert x_epsilon_r < c < x_epsilon_l
        print(f"corresponding x interval: {x_epsilon_r} < {c} < {x_epsilon_l}")
    except AssertionError:
        print('Error! Cannot find the x epsilon boundary...')
        print(f"coresponding x interval: {x_epsilon_r} < {c} < {x_epsilon_l}")
        print(f"Limit does not exist")
        exit(1)

    # Determine how far epsilon x is from c
    distance_from_epsilon_x_to_c_r = c - x_epsilon_r
    distance_from_epsilon_x_to_c_l = x_epsilon_l - c
    print(f"distance from {x_epsilon_l} to {c} (left): {distance_from_epsilon_x_to_c_l}")
    print(f"distance from {x_epsilon_r} to {c} (right): {distance_from_epsilon_x_to_c_r}")

    # Get the min value (allows for symmetry) :
    min_value = min(distance_from_epsilon_x_to_c_r, distance_from_epsilon_x_to_c_l)
    print(f"Lets take the min value: {min_value}\n")

    print(f"Scratch work complete...")
    print(f"We've shown that for epsilon: {epsilon}")
    print(f"If we have {x_epsilon_r} < {c} < {x_epsilon_l}")
    print(f"then we have  {epsilon_r} < {limit} < {epsilon_l}")
    print(f"We can now choose a delta < {min_value} and be sure our x and f(x) values will be within our epsilon values\n")

    print(f"Lets sample some delta values < {min_value}\n")
    delta = min_value
    try:
        for i in range(0, 5):
            print(f"let delta be: {delta}")
            assert c - delta < c < c + delta
            assert math.sqrt(c - delta) < limit < math.sqrt(c + delta)
            print(f"x is within: {c - delta} < {c} < {c + delta} ")
            print(f"f(x) is within: {math.sqrt(c - delta)} < {limit} < {math.sqrt(c + delta)}  \n")
            # Halve the delta each time
            # Cast to float to avoid scientific notation
            delta = float(delta * 1 / 2)
    except AssertionError:
        print('Error! Boundary loss!')
        print(f"x is within: {c - delta} < {c} < {c + delta} ")
        print(f"f(x) is within: {math.sqrt(c - delta)} < {limit} < {math.sqrt(c + delta)}  \n")
        print(f"Limit does not exist")
        exit(1)


    return True

epsilon = [1, .5, .1, .01, .001, .0001]
limit = 2
c = 4
for e in epsilon:
    limit_exists = epsilon_delta_sqrt(c=4, limit=2, epsilon=float(e))
    if limit_exists:
        print(f'limit exists for f(x) = sqrt(x), when x approaches {c}, then the limit is {limit}')
    else:
        print(f'limit does not exists for f(x) = sqrt(x), when x approaches {c}, then the limit is not {limit}')

