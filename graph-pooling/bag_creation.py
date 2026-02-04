import math
import random
import torch
from utils import get_flattened_nodes

def get_eval_nodes(m, n, chosen_label, labels, label_node, label_non_node, bag_pos_ratio, seed=42):
    bag_node = dict()

    # internal node ratio for positive and negative bags
    bag_neg_ratio = 1 - bag_pos_ratio

    pos_count = 0
    neg_count = 0

    for i in range(m):  # m = number of bags
        # randomly decide if the bag is positive or negative
        make_positive = random.random() < 0.5  # 50% chance

        if make_positive:
            p = math.ceil(n * bag_pos_ratio)  # number of positive nodes
        else:
            p = math.ceil(n * bag_neg_ratio)  # number of positive nodes for negative bag

        q = n - p  # remaining nodes are negative

        # select nodes
        selected_node = random.sample(label_node[chosen_label], p) + \
                        random.sample(label_non_node[chosen_label], q)

        # shuffle nodes inside the bag
        random.shuffle(selected_node)

        # assign bag label based on intended positive/negative
        positive = 1 if make_positive else 0

        bag_node[tuple(selected_node)] = positive

        # update counters
        if positive == 1:
            pos_count += 1
        else:
            neg_count += 1

    print(f"Mode 3: Number of positive bags = {pos_count}, Number of negative bags = {neg_count}")

    node_train = get_flattened_nodes(bag_node, chosen_label, labels, 0, 300)
    node_eva = get_flattened_nodes(bag_node, chosen_label, labels, 300, 500)

    b = len(node_train)
    c = len(node_eva)
    d = int(0.5 * c)
    idx_val = torch.LongTensor(range(d))
    idx_test = torch.LongTensor(range(d, c))
    idx_train = torch.LongTensor(range(b))

    return idx_train, idx_val, idx_test, node_train, node_eva