import numpy as np
from util import compute_jaccard
import data_bitcoin as btc
import data_auto_sys as aus
import data_UCI as uci

def avg_jaccard(data):
    jaccard = []
    edges = data.edges['idx']
    time_step = np.unique(edges[:, 2])
    time_step = np.sort(time_step)
    for t in time_step[:-1]:
        graph_1 = edges[edges[:, 2]==t, :2]
        graph_2 = edges[edges[:, 2]==t+1, :2]
        jaccard.append(compute_jaccard(graph_1, graph_2))
    return jaccard

if __name__ == '__main__':
    data_alpha = btc.Bitcoin_Dataset('data/BTC-ALPHA/')
    j_alpha = avg_jaccard(data_alpha)
    print(np.mean(j_alpha))

    data_otc = btc.Bitcoin_Dataset('data/BTC-OTC/')
    j_otc = avg_jaccard(data_otc)
    print(np.mean(j_otc))

    data_uci = uci.UC_Irvine_Dataset('data/UCI/')
    j_uci = avg_jaccard(data_uci)
    print(np.mean(j_uci))

    data_as= aus.Autonomous_System_Dataset('data/AS-733/')
    j_as = avg_jaccard(data_as)
    print(np.mean(j_as))
