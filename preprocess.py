# coding=utf-8
import datetime
import pickle
import random

#  Parameters
random.seed(0)
dt = 3600  #
request_time = 12 * 3600  # request 12 hrs ago
threshold_dist = 0.03  # if detour is larger than threshold, driver and passenger cannot be matched.

# open files
f = open('taxi_csv1_1.pkl', 'rb')
data = pickle.load(f)
f.close()
# print(data.tail(5))
# print(len(data))
# only take 30000 records
data = data.head(30000)


# from datetime to timestamp
def getstamp(s):
    time = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return time.timestamp() - 1356969600 + request_time


L_index = []  # index of set L or driver
R_index = []  # index of set R or passenger

graphic_data = []
# generate vertex type and request time t0
for i in range(len(data)):
    singledata = []
    vertex_type = random.randint(0, 1)
    singledata.append(vertex_type)
    cap = data.iloc[i][' passenger_count']
    if vertex_type == 0:
        L_index.append(i)
        # to ensure the cap is not very small
        if cap <= 3:
            cap = random.randint(3, 6)
    else:
        R_index.append(i)
    singledata.append(cap)
    t1 = getstamp(data.iloc[i][' pickup_datetime'])
    # print(t1)
    singledata.append(t1 - request_time)
    # singledata.append(str(getstamp(data.iloc[i][' pickup_datetime'])))
    # singledata.append(str(getstamp(data.iloc[i][' dropoff_datetime'])))
    # singledata.append(str(data.iloc[i][' passenger_count']))
    # singledata.append(str(data.iloc[i]['pickup_coordinates'][0]))
    # singledata.append(str(data.iloc[i]['pickup_coordinates'][1]))
    # singledata.append(str(data.iloc[i]['dropoff_coordinates'][0]))
    # singledata.append(str(data.iloc[i]['dropoff_coordinates'][1]))
    graphic_data.append(singledata)


# In[83]:


# distance between two points
# latitude ~ 40.7
# cos40.7 = 0.758

def dist(pt1, pt2):
    return 0.758 * abs(pt1[0] - pt2[0]) + abs(pt1[1] - pt2[1])


# In[94]:


# check if a driver and a passenger can match
def check(data, graphic_data, l, r):
    # check capacity

    if graphic_data[l][1] < graphic_data[r][1]:
        return False

    # check pickup time

    t_l = getstamp(data.iloc[l][' pickup_datetime'])
    t_r = getstamp(data.iloc[r][' pickup_datetime'])
    #     print([t_l-dt, t_l+dt], [t_r-dt, t_r+dt])
    if ((t_l + dt) < (t_r - dt)) or ((t_r + dt) < (t_l - dt)):
        return False

    # check de-tour
    # from "a->b" to "a->c->d->b"
    # some data has 0 for its coordinates which will be isolated in the generated graph
    # TO-DO: clear these "0" data from very beginning
    a = data.iloc[l]['pickup_coordinates']
    b = data.iloc[l]['dropoff_coordinates']
    c = data.iloc[r]['pickup_coordinates']
    d = data.iloc[r]['dropoff_coordinates']
    detour = dist(a, c) + dist(c, d) + dist(d, b) - dist(a, b)
    #     print(detour)
    if detour > threshold_dist:
        return False
    # print(a, b, c, d, detour)
    return True
    pass


# In[85]:


# build graph
edges = []
for i in range(len(L_index)):
    for j in range(len(R_index)):
        l = L_index[i]
        r = R_index[j]
        link_ij = check(data, graphic_data, l, r)
        if link_ij:
            graphic_data[l].append(r)
            graphic_data[r].append(l)
            edges.append([l, r])

# In[86]:


with open('edge.txt', 'w') as f:
    for e in edges:
        f.write(str(e[0]) + ' ' + str(e[1]) + '\n')

# In[87]:


with open('adj.txt', 'w') as f:
    for v in graphic_data:
        strv = ''
        for i in v:
            strv += str(i)
            strv += ' '
        f.write(strv + '\n')

# In[91]:


print('number of edges:', len(edges))

# In[93]:


print('number of vertices:', len(graphic_data))

