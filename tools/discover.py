import numpy as np
import joblib
import random
import copy
import pickle
from scipy.special import softmax

np.random.seed(1)

num_samples = 50
iters = 50
tolerance = 0.0001
lr = 0.001

# - load dataset
savedir = "save"
tr_x = np.load("./matpred/{}/train_data.npy".format(savedir))
tr_y = np.load("./matpred/{}/train_labels.npy".format(savedir))

o_cat_idxs = np.array([2,5,11,12,14,17,19,21,23,25,28])
o_num_idxs = np.array([3,4,6,13,15,16,18,20,22,24,26,27,29,30,31,32,33])
cat_idxs = list(range(len(o_cat_idxs)))
num_idxs = [len(o_cat_idxs) + i for i in list(range(len(o_num_idxs)))]
feat_idxs = list(range(len(cat_idxs)+len(num_idxs)))
catmap = {}
for idx in cat_idxs:
    catmap[idx] = list(set(tr_x[:,idx]))
for k,v in catmap.items():
    catmap[k] = [int(vv) for vv in v]
del(catmap[2][-1])
del(catmap[3][-1])
del(catmap[9][-1])
nummap = {}
for idx in num_idxs:
    nummap[idx] = np.std(tr_x[:,idx])
print("catmap")
print(catmap)
print("nummap")
print(nummap)

# - load pred model(s)
model_comp = joblib.load("./matpred/{}/pred_model_0.joblib".format(savedir))
model_flex1 = joblib.load("./matpred/{}/pred_model_1.joblib".format(savedir))

def rearrange(s):
    o_idxs = np.concatenate((o_cat_idxs, o_num_idxs), 0)
    temp_s = ["none" for _ in range(37)]
    for i,v in enumerate(s):
        if i >= len(o_idxs):
            break
        temp_s[o_idxs[i]] = v
    temp_s = np.concatenate((temp_s, s[-4:]), 0)
    return temp_s

def coalesce(s):
    temp_s = copy.deepcopy(s)
    offset = 11
    if s[1] == s[0]:
        temp_s[offset+0] += temp_s[offset+2]
        temp_s[1] = -1.
        temp_s[offset+2] = 0.
    if s[4] == s[3]:
        temp_s[offset+3] += temp_s[offset+4]
        temp_s[4] = -1.
        temp_s[offset+4] = 0.
    for o in [6,7,8,9]:
        if s[o-1] == s[3]:
            temp_s[offset+3] += temp_s[offset+o]
            temp_s[o-1] = -1.
            temp_s[offset+o] = 0.
    for o in [6,7,8,9]:
        if s[o-1] == s[4]:
            temp_s[offset+4] += temp_s[offset+o]
            temp_s[o-1] = -1.
            temp_s[offset+o] = 0.
    for o in [7,8,9]:
        if s[o-1] == s[5]:
            temp_s[offset+6] += temp_s[offset+o]
            temp_s[o-1] = -1.
            temp_s[offset+o] = 0.
    for o in [8,9]:
        if s[o-1] == s[6]:
            temp_s[offset+7] += temp_s[offset+o]
            temp_s[o-1] = -1.
            temp_s[offset+o] = 0.
    if s[8] == s[7]:
        temp_s[offset+8] += temp_s[offset+9]
        temp_s[8] = -1.
        temp_s[offset+9] = 0.
    return temp_s

def distribute_to_one(s):
    o_idxs = [i+11 for i in [0,2,3,4,5,6,7,8,9,10]]
    temp_s = copy.deepcopy(s)
    vals = []
    idxs = []
    for i in o_idxs:
        if s[i] > 0.:
            vals.append(s[i])
            idxs.append(i)
    new_vals = softmax(np.array(vals)/0.18)
    for i in range(len(new_vals)):
        diff = new_vals[i] - vals[i]
        vals[i] += diff/2.
    for ii,i in enumerate(idxs):
        temp_s[i] = vals[ii]
    return temp_s

# cost fxn
def cost(s):
    # - pred strength
    str_comp = max(0., model_comp.predict(np.expand_dims(np.concatenate((s[feat_idxs], [0]), 0), 0))[0])
    str_flex1 = max(0., model_flex1.predict(np.expand_dims(np.concatenate((s[feat_idxs], [0]), 0), 0))[0])

    # - pred carbon/cost
    v_carb = max(0., model_carb.predict(np.expand_dims(s[feat_idxs], 0))[0])
    v_cost = max(0., model_cost.predict(np.expand_dims(s[feat_idxs], 0))[0])

    r_cco = v_cost / str_comp
    r_cca = v_carb / str_comp
    r_fco = v_cost / str_flex1
    r_fca = v_carb / str_flex1

    zero_cost = 0.
    for ni in num_idxs[:-6]:
        if 0. == s[ni]:
            zero_cost += 0.4

    # (note) water %weight / total binder %weight ratio (> 0.7, < 0.3 unrealistic)
    ratio_cost = 0.
    water_pw = s[16]
    b1_pw = s[13]
    b2_pw = s[15]
    if 0. == (b1_pw + b2_pw):
        ratio_cost = 2.
    else:
        wb_ratio = water_pw / (b1_pw + b2_pw)
        if wb_ratio > 0.7 or wb_ratio < 0.3:
            ratio_cost += 2.

    cost = 0

    w1 = 5.0
    w2 = 5.0
    w3 = 5.0
    w4 = 5.0
    cost = w1*r_cco + w2*r_cca + w3*r_fco + w4*r_fca + zero_cost + ratio_cost

    return cost, [r_cco, r_cca, r_fco, r_fca]

def opt(s, c, ss, cs, catws, freqws, iteration):
    assert len(ss) == len(cs)
    assert (s == ss[-1]).all()
    assert c == cs[-1]

    prob = np.random.rand()
    if 0 == iteration or prob > 0.99:
        # random mutate
        if 0 == iteration:
            mut_prob = 0.5
        else:
            mut_prob = 0.85
        for idx in cat_idxs:
            prob = np.random.rand()
            if prob > mut_prob:
                mut_val = np.random.choice(catmap[idx], p=softmax(np.array(freqws[idx])/100.))
                s[idx] = float(mut_val)
                mut_prob += 0.15
                mut_prob = min(mut_prob, 0.95)
            else:
                mut_prob -= 0.15
                mut_prob = max(mut_prob, 0.05)
        for idx in num_idxs:
            prob = np.random.rand()
            if prob > mut_prob:
                mut_val = np.random.uniform(-nummap[idx],nummap[idx])
                s[idx] = max(s[idx] + mut_val, 0.)
                mut_prob += 0.15
                mut_prob = min(mut_prob, 0.95)
            else:
                mut_prob -= 0.15
                mut_prob = max(mut_prob, 0.05)

        return s
    else:
        last_s = ss[-2]
        last_c = cs[-2]

        c_diff = c - last_c
        s_diff = s - last_s

        seen1 = {}
        seen2 = {}
        seen3 = {}
        for idx in cat_idxs:
            prob = np.random.rand()
            if c_diff > 1. and prob > 0.7:
                ws = catws[idx]
                ws = [1./temp_w for temp_w in ws]
                sel = np.random.choice(catmap[idx], p=softmax(np.array(ws)/1.5))
                s[idx] = sel
        for idx in num_idxs:
            s[idx] = max(s[idx] + (s_diff[idx] * -c_diff * 0.5), 0.)

        return s

def run():
    idxs = np.random.randint(0, len(tr_x), num_samples)
    samples = [[copy.deepcopy(tr_x[idx])] for idx in idxs]
    sample_costs = [[] for _ in range(len(samples))]
    sample_costcomps = [[] for _ in range(len(samples))]
    done = [False for _ in range(len(samples))]
    history = 2
    best = []
    best_vals = 10000.

    # init cat vals, shape should be [num cats, num unique]
    catws = []
    freqws = []
    catidxmap = {}
    for k,v in catmap.items():
        catidxmap[k] = {}
        catws.append([])
        freqws.append([])
        for i,vv in enumerate(v):
            catidxmap[k][vv] = i
            catws[-1].append([])
            freqws[-1].append(0)
    for i,d in enumerate(tr_x):
        samp_cost, cost_comps = cost(d)
        for j in cat_idxs:
            v = d[j]
            if v in catidxmap[j]:
                catws[j][catidxmap[j][v]].append(samp_cost)
                freqws[j][catidxmap[j][v]] += 1
    for j in range(len(catws)):
        for k in range(len(catws[j])):
            catws[j][k] = np.mean(catws[j][k])
    for j in range(len(freqws)):
        if j not in [2,3,9]:
            freqws[j][-1] = max(freqws[j][:-1])

    best_samples_overall = copy.deepcopy(samples)
    best_samples_costs = [10000. for _ in range(len(samples))]
    best_samples_costcomps = [[] for _ in range(len(samples))]

    for j in range(iters):
        print("iter: {}".format(j))
        print("--------")
        for i in range(len(samples)):
            if done[i]:
                continue
            samp = samples[i][-1]
            # - calc cost
            samp_cost, cost_comps = cost(samp)
            sample_costs[i].append(samp_cost)
            sample_costcomps[i].append(cost_comps)
            if samp_cost <= best_vals:
                best_vals = samp_cost
                best.append([samp, samp_cost])
            if samp_cost < best_samples_costs[i]:
                best_samples_costs[i] = samp_cost
                best_samples_costcomps[i] = cost_comps
                best_samples_overall[i] = copy.deepcopy(samp)
            # - opt step
            upd = opt(copy.deepcopy(samp), samp_cost, samples[i], sample_costs[i], catws, freqws, j)
            step = copy.deepcopy(upd)
            for k in range(len(upd)):
                if k < 11:
                    step[k] = 0.
                else:
                    step[k] = -lr * (samp[k] - step[k])
            if j > 1000 and np.all(np.abs(step) <= tolerance):
                print("--- DONE ---")
                done[i] = True
            else:
                upd_samp = copy.deepcopy(samp)
                for k in range(len(step)):
                    if k < 11:
                        upd_samp[k] = upd[k]
                    else:
                        upd_samp[k] += step[k]
                if history == len(samples[i]):
                    np.delete(samples[i], 0, 0)
                    np.delete(sample_costs[i], 0, 0)
                    np.delete(sample_costcomps[i], 0, 0)
                upd_samp = coalesce(upd_samp)
                upd_samp = distribute_to_one(upd_samp)
                samples[i] = np.concatenate((samples[i], np.expand_dims(copy.deepcopy(upd_samp), 0)), 0)
                if j > 0:
                    for k in cat_idxs:
                        v = upd_samp[k]
                        cost_diff = samp_cost - sample_costs[i][-2]
                        if v in catidxmap[k] and v in catws[k]:
                            catws[k][catidxmap[k][v]] = catws[k][catidxmap[k][v]] + lr * cost_diff
        print([sc[-1] for sc in sample_costs])

    print("final samples")
    temp_samples = []
    for si,s in enumerate(best_samples_overall):
        print(best_samples_costs[si])
        temp_samples.append(s)

    # convert to interpretable table (with header row, cat labels, str, carb, cost)
    with open("./matpred/header_map.txt", 'r') as f:
        header_str = f.read()
    headers = header_str.split(",")
    with open("./matpred/info_map.pkl", 'rb') as f:
        all_info = pickle.load(f)

    final_samples = []
    final_samples.append(headers[:-9] + ["Ratio (Cost-Compressive)", "Ratio (Carbon-Compressive)", "Ratio (Cost-Flexural)", "Ratio (Carbon-Flexural)"])
    for si,s in enumerate(best_samples_overall):
        temp_samp = []
        for sj in range(len(s)):
            if sj < len(cat_idxs):
                if -1 == int(s[sj]):
                    temp_samp.append("none")
                else:
                    sj_idx = sj if sj < 2 else sj+1
                    temp_samp.append(all_info[sj_idx][int(s[sj])])
            else:
                temp_samp.append(s[sj])
        temp_samp = temp_samp + best_samples_costcomps[si]
        final_samples.append(temp_samp)

    # rearrange to default
    for i in range(len(final_samples)):
        final_samples[i] = rearrange(final_samples[i])

    # fix:
    # - remove none -> >0 % and vice versa
    for i in range(len(final_samples)):
        if 0 == i:
            continue
        fs = final_samples[i]
        for j in o_cat_idxs:
            if "none" == fs[j] and 11 != j:
                fs[j+1] = 0.
            if 11 != j and 0. == float(fs[j+1]):
                fs[j] = "none"
    final_samples = np.delete(final_samples, 11, 1)

    np.savetxt("./db/discover_samples.csv", final_samples, delimiter=",", fmt='%s')

if __name__ == "__main__":
    run()
