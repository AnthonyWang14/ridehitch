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

#check match only for space and time
def check_match2(supply, demand, T_threshold, D_threshold):
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
    return np.sqrt(np.square(s[0]-d[0]) + np.square(s[1]-d[1]))

def weight(sup, dem):
    dis1 = dist([sup[sx_idx], sup[sy_idx]], [dem[sx_idx], dem[sy_idx]])
    dis2 = dist([dem[sx_idx], dem[sy_idx]], [dem[dx_idx], dem[dy_idx]])
    dis3 = dist([dem[dx_idx], dem[dy_idx]], [sup[dx_idx], sup[dy_idx]])
    dis4 = dist([sup[sx_idx], sup[sy_idx]], [sup[dx_idx], sup[dy_idx]])
    weight = (dis1 + dis2 + dis3) / 10
    return weight

def bounded_normal(loc, std_, low, high):
    x = np.random.normal(loc, std_)
    x = round(x)
    if x < low:
        x = low
    if x > high - 1:
        x = high - 1
    return x
