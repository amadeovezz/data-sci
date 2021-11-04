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


def epsilon_delta_sqrt(c: float, limit: float) -> bool:
    """
    numeric epsilon delta solver for f(x) = sqrt(x)
    determines whether the limit l exists, as x -> c = l

    symbolically and with algebra gives you a generic answer.
    numerically doesn't.

    """

    def get_delta_in_terms_of_epsilon(c: float, limit: float, epsilon: float) -> float:
        # Find our epsilon boundary, this can be any epsilon
        # | f(x) - L | < e = L - e < f(x) < L + e

        print(f"Performing scratch work, assuming p->q (the limit exists) looking for delta values...")
        print(f"Using some epsilon values to get corresponding delta values...\n")
        epsilon_l = limit + epsilon
        epsilon_r = limit - epsilon
        print(f"(L - epsilon, epsilon + L) interval: {epsilon_r} < {limit} < {epsilon_l}")

        # Find the inverse of f(x) = sqrt(x) , which is just f(x) = x^2
        delta_left = epsilon_l ** 2
        delta_right = epsilon_r ** 2

        try:
            assert delta_right < c < delta_left
            print(f"corresponding (delta - c, delta + c) interval: {delta_right} < {c} < {delta_left}")
        except AssertionError:
            print('Error! Cannot find the delta boundary...')
            print(f"(delta - c, delta + c) interval: {delta_right} < {c} < {delta_left}")
            print(f"Limit does not exist")
            exit(1)

        # Now find a symmetric delta
        print(f"looking for delta between: {delta_right} < {c} < {delta_left}")

        # Delta must be less than
        distance_from_delta_to_c_r = c - delta_right
        distance_from_delta_to_c_l = delta_left - c
        print(f"distance from {delta_left} to {c} (left): {distance_from_delta_to_c_l}")
        print(f"distance from {delta_right} to {c} (right): {distance_from_delta_to_c_r}")

        # Min value (allows for symmetry) :
        min_value = min(distance_from_delta_to_c_r, distance_from_delta_to_c_l)
        print(f"Lets take the min value: {min_value}\n")
        print(f"Our delta now must be < {min_value}")
        return min_value

    def check_delta(c: float, limit: float, min_value: float):
        delta = min_value
        print(f"Sampling some values < {min_value}\n")
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
            print('Error! Boundary error!')
            print(f"x is within: {c - delta} < {c} < {c + delta} ")
            print(f"f(x) is within: {math.sqrt(c - delta)} < {limit} < {math.sqrt(c + delta)}  \n")
            print(f"Limit does not exist")
            exit(1)

    print(f"Verifying lim x->{c} sqrt(x) = {limit}\n")
    epsilon = [1, .5, .1, .01, .001, .0001]
    for e in epsilon:
        print(f"let epsilon be: {e}")
        min_value = get_delta_in_terms_of_epsilon(c, limit, float(e))
        print(f"Now that we have our min_value: {min_value}, lets check p -> q")
        check_delta(c, limit, min_value)

    return True


epsilon_delta_sqrt(4, 2)
