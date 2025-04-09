"""
Various helpful mathematical functions.
"""

def interpolate(ts, xs, t):
    """
    First-order interpolation of a time series (ts, xs) at time t

    Args:
        ts (_type_): Sampling times
        xs (_type_): Sampling values
        t (_type_): time to interpolate at
    """
    if not ts[0] <= t <= ts[-1]:
        raise ValueError("t is not within ts")
    
    for i, time in enumerate(ts):
        if time > t: break
    
    return (t - ts[i-1])/(ts[i] - ts[i-1]) * (xs[i] - xs[i-1]) + xs[i-1]