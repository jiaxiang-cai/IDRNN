import random
import numpy as np
# l = 100
# seq = []
# for i in range(0,l):
#     aa = random.randint(0,19)
#     seq. append (aa)
# seq = [0, 3, 11, 11, 9, 1, 12, 7, 13, 7, 18, 15, 18, 12, 10, 12, 10, 4, 13, 4, 1, 13, 2, 5, 6, 0, 2, 3, 0, 14, 10, 0,
#        8, 11, 18, 16, 1, 16, 15, 0, 18, 0, 11, 18, 10, 13, 2, 11, 3, 6, 2, 8, 7, 18, 10, 5, 18, 12, 6, 16, 6, 6, 6, 15,
#        19, 5, 5, 14, 7, 6, 5, 3, 13, 11, 9, 16, 0, 12, 10, 2, 11, 4, 2, 12, 15, 7, 18, 8, 4, 5, 16, 13, 11, 8, 18, 4, 5,
#        16, 4, 0]


def count_seq(seq):
    # Reference
    reference = {}
    prob = [0.052187985, 0.011206257, 0.060427483, 0.057636033, 0.041804329, 0.040064729, 0.024772436, 0.062504214,
            0.065093385, 0.097296204, 0.023626188, 0.095354325, 0.058674398, 0.048196345, 0.04037489, 0.12406446,
            0.066913897, 0.044514868, 0.006890972, 0.030584586]

    for i in range(20):
        reference[i] = prob[i]
    l = len(seq)
    summary={}
    counted = {key: 0 for key in range(20)}
    # print(all_aa)
    unique_elements, counts = np.unique(seq, return_counts=True)
    for i in range(len(unique_elements)):
        counted[unique_elements[i]] = counts[i] / l

    summary['counted']=counted
    #Comparision

    diff = {}
    for aa in range(20):
        diff[aa]= counted[aa] - reference[aa]
    sorted_diff = sorted(diff.items(), key=lambda x: x[1], reverse=True)
        # Create a new dictionary from the sorted list of items
    sorted_diff = dict(sorted_diff)

    summary['sorted_diff']=sorted_diff
    return sorted_diff
def conditioned_seq(seq):
    sorted_diff = count_seq(seq)
    less_aa = []
    more_aa = []
for aa in sorted_diff:
    if sorted_diff[aa]>0.03:
        less_aa.append(aa)
    if sorted_diff[aa]<-0.03:
        more_aa.append(aa)
    # print('before', count_seq(seq))
    for aa in less_aa:
        all_pos_aa = [index for index, element in enumerate(seq) if element == aa]
        print(aa,all_pos_aa)
        pos_to_change = random.sample(all_pos_aa, int(sorted_diff[aa]*l))
        print(pos_to_change)
        for pos in pos_to_change:
            seq[pos] = random.choice(more_aa)

    # print(seq)
    # print('after',count_seq(seq))
    return seq
