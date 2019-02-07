import numpy as np


request_len = 7
type_idx = 0
time_idx = 1
sx_idx = 2
sy_idx = 3
dx_idx = 4
dy_idx = 5
cap_idx = 6
idx_request_idx = 7


# check match
def check_match(supply, demand, T_threshold, D_threshold):
    if supply[cap_idx] < demand[cap_idx]:
        return False
    if np.abs(supply[time_idx] - demand[time_idx]) > T_threshold:
        return False
    a = [supply[sx_idx], supply[sy_idx]]
    b = [supply[dx_idx], supply[dy_idx]]
    c = [demand[sy_idx], demand[sy_idx]]
    d = [demand[dx_idx], demand[dy_idx]]
    old_path = dist(a, b)
    new_path = dist(a, c) + dist(c, d) + dist(d, b)
    detour = new_path - old_path
    if detour > D_threshold:
        return False
    return True


def dist(s, d):
    return np.abs(s[0] - d[0]) + np.abs(s[1] - d[1])


def bounded_normal(loc, std_, low, high):
    x = np.random.normal(loc, std_)
    x = round(x)
    if x < low:
        x = low
    if x > high - 1:
        x = high - 1
    return x
