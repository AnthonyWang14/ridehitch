from utilities import *

T_threshold = 50
D_threshold = 50
state_pool_size = 50
if __name__ == '__main__':
    adj_list = []
    requests_list = []
    filename = 'data/norm1000.txt'
    with open(filename, 'rt') as f:
        idx = 0
        supply_pool = []
        for line in f:
            d = [int(i) for i in line.strip().split()]
            adjs = []
            if d[type_idx] == 0:
                supply_pool.append(idx)
            if d[type_idx] == 1:
                for i in supply_pool:
                    if requests_list[i][type_idx] == 0 and check_match(requests_list[i], d, T_threshold, D_threshold):
                        adjs.append(i)
                        if len(adjs) > state_pool_size:
                            adjs.pop(0)
                    for j in adjs:
                        adj_list[j].append(idx)
            adj_list.append(adjs)
            idx += 1
            requests_list.append(d)
    request_num = len(requests_list)
    print(len(adj_list))
    filename_adj = filename+'T'+str(T_threshold)+"D"+str(D_threshold)+"P"+str(state_pool_size)
    gulidian_count = 0
    for i in range(len(adj_list)):
        if len(adj_list[i]) == 0 and requests_list[i][type_idx] == 0:
            gulidian_count += 1
    print('gulidiangeshu',  gulidian_count)
    with open(filename_adj, 'w') as f:
        for i in range(len(adj_list)):
            str_arr = [str(item) for item in adj_list[i]]
            print(str(requests_list[i][type_idx])+' '+str(requests_list[i][cap_idx]) + ' '+ " ".join(str_arr), file=f)

