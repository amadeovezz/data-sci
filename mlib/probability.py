import pandas as pd

from scipy import special


def bayes(intersection: float, total: float) -> float:
    """
    An implementation of bayes theorem using counts.
    :param intersection:
    :param total:
    :return:
    """
    cond_prob = (
            intersection / total
    )
    rounded_cond_prob = round(cond_prob, 4)
    assert 0 < rounded_cond_prob < 1
    return rounded_cond_prob


def bin_pdf(n: int, x: int, p: float) -> float:
    """
    The binomial pdf.
    :param n: number of trials
    :param x: number of successful trials
    :param p: probability of success
    :return:
    """
    if not 0 <= p <= 1:
        raise AssertionError(f'p must be: 0 <= p <= 1, p is: {p}')
    return round(special.comb(n, x) * p ** x * (1 - p) ** (n - x), 5)


def binom_dist(rv: str, num_of_trials: int, success: int) -> pd.DataFrame:
    """

    Creates a binomial distribution.

    Example usage:
    df = probability.binom('num_of_heads', 100, .4)
    sns.relplot(data=df,x='num_of_heads', y='probability')

    :param rv: name of the random variable, ex: num_of_heads
    :param num_of_trials: how many independent trials are in the experiment
    :param success: real number between 0 < success < 1
    :return:
    """
    probability = [bin_pdf(num_of_trials, x, success) for x in range(0, num_of_trials)]
    assert .95 < sum(probability) <= 1
    d = {
        rv: [i for i in range(0, num_of_trials)],
        'probability': probability
    }
    return pd.DataFrame(data=d)


def binom_dist_likelihood(num_of_trials: int, num_of_success: int) -> pd.DataFrame:
    """
    :param num_of_trials: how many independent trials are in the experiment
    :param num_of_success: observed successes, ex: X=7
    :return:
    """
    values_of_p = [x / 100 for x in range(1, 100)]
    d = {
        'p': [p for p in values_of_p],
        'likelihood': [bin_pdf(num_of_trials, num_of_success, p) for p in values_of_p]
    }
    return pd.DataFrame(data=d)
