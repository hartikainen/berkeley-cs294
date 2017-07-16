import scipy

def discount(x, gamma):
    """
    Compute discounted sum of future values

    [https://github.com/berkeleydeeprlcourse/homework/blob/master/hw4/main.py]
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
