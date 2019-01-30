import numpy as np

# check match
def check_match(supply, demand, T_threshold, D_threshold):
    if supply[6] < demand[6]:
        return False
    if np.abs(supply[1]-demand[1]) > T_threshold:
        return False
    a = [supply[2], supply[3]]
    b = [supply[4], supply[5]]
    c = [demand[2], demand[3]]
    d = [demand[4], demand[5]]
    old_path = dist(a,b)
    new_path = dist(a,c) + dist(c,d) + dist(d,b)
    detour = new_path - old_path
    if detour > D_threshold:
        return False
    return True

def dist(s, d):
    return np.abs(s[0]-d[0]) + np.abs(s[1]-d[1])


def bounded_normal(loc, std_, low, high):
	x = np.random.normal(loc,std_)
	x = round(x)
	if x < low:
		x = low
	if x > high-1:
		x = high-1
	return x
