from utilities import *

T_threshold = 50
D_threshold = 50

if __name__ == '__main__':
    adj_list = []
    requests_list = []
    filename = 'data/norm100.txt'
    with open(filename, 'rt') as f:
        idx = 0
        for line in f:
            d = [int(i) for i in line.strip().split()]
            adjs = []
            if d[type_idx] == 1:
                for i in range(len(requests_list)):
                    if check_match(requests_list[i], d, T_threshold, D_threshold):
                        adjs.append(i)
                        adj_list[i].append(idx)
            adj_list.append(adjs)
            idx += 1
            requests_list.append(d)
    request_num = len(requests_list)
    print(len(adj_list))
    filename_adj = filename+'T'+str(T_threshold)+"D"+str(D_threshold)+".txt"
    with open(filename_adj, 'w') as f:
        for i in range(len(adj_list)):
            str_arr = [str(item) for item in adj_list[i]]
            print(str(requests_list[i][cap_idx]) + ' '+ " ".join(str_arr), file=f)

