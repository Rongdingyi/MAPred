import random

def mine_hard_negative(dist_map, knn=10):
    #print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    for i, target in enumerate(ecs):
        sort_orders = sorted(
            dist_map[target].items(), key=lambda x: x[1], reverse=False)
        if sort_orders[1][1] != 0:
            freq = [1/i[1] for i in sort_orders[1:1 + knn]]
            neg_ecs = [i[0] for i in sort_orders[1:1 + knn]]
        elif sort_orders[2][1] != 0:
            freq = [1/i[1] for i in sort_orders[2:2+knn]]
            neg_ecs = [i[0] for i in sort_orders[2:2+knn]]
        elif sort_orders[3][1] != 0:
            freq = [1/i[1] for i in sort_orders[3:3+knn]]
            neg_ecs = [i[0] for i in sort_orders[3:3+knn]]
        else:
            freq = [1/i[1] for i in sort_orders[4:4+knn]]
            neg_ecs = [i[0] for i in sort_orders[4:4+knn]]

        normalized_freq = [i/sum(freq) for i in freq]
        negative[target] = {
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative

def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec = id_ec[anchor]
    pos_ec = random.choice(anchor_ec)
    neg_ec = mine_neg[pos_ec]['negative']
    weights = mine_neg[pos_ec]['weights']
    result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    while result_ec in anchor_ec:
        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    neg_id = random.choice(ec_id[result_ec])
    return neg_id


# def random_positive(id, id_ec, ec_id):
#     pos_ec = random.choice(id_ec[id])
#     pos = id
#     if len(ec_id[pos_ec]) == 1:
#         return pos + '_' + str(random.randint(0, 29))
#     elif len(ec_id[pos_ec]) == 2:
#         return pos + '_' + str(random.randint(0, 24))
#     elif len(ec_id[pos_ec]) == 3:
#         return pos + '_' + str(random.randint(0, 19))
#     elif len(ec_id[pos_ec]) == 4:
#         return pos + '_' + str(random.randint(0, 14))
#     elif len(ec_id[pos_ec]) == 5:
#         return pos + '_' + str(random.randint(0, 9))
#     while pos == id:
#         pos = random.choice(ec_id[pos_ec])
#     return pos

def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    pos = id
    if len(ec_id[pos_ec]) == 1:
        return pos + '_' + str(random.randint(0, 9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    return pos