from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
import joblib
import pickle
import copy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import GridSearchCV

np.random.seed(0)

db_path = "../db/db_postproc.npy"

do_analysis = False
featpick = 0
do_hparam = True #False

cat_idxs = np.array([2,5,8,11,12,14,17,19,21,23,25,28])
mat_idxs = np.array([3,4,6,7,9,10,13,15,16,18,20,22,24,26,27,29,30,31,32,33])
str_idxs = np.array([37,38,39,40,41,42,43,44,45])

def load_data():
    print("loading: candidate db - {}".format(db_path))
    db = np.load(db_path)

    # remove no-agg and no-binder
    idxs = [0]
    for i in range(len(db)):
        if 0 == i:
            continue
        d = db[i]
        keep = True
        # check agg
        if "none" == d[25]:
            keep = False
        # check binders
        if "none" == d[12] and "none" == d[14]:
            keep = False
        if keep:
            idxs.append(i)
    db = db[idxs]

    for i in range(db.shape[0]):
        v = db[i][11]
        n_v = db[i][12]
        if "none" == v and "none" != n_v:
            db[i][11] = n_v

    # NOTE need to upload cost and carbon footprint data first
    cc_db = pd.read_csv("../db/ccvals.csv", header=None, encoding="utf_8").to_numpy()
    cc_db = cc_db[idxs]

    return db, cc_db

def clean(db):
    # + convert
    f_db = np.zeros(db.shape)
    all_revinfo = {}
    for j in range(12):
        info = {}
        revinfo = {}
        track = 0
        for i in range(db.shape[0]):
            v = db[i,j].strip().lower()
            if v != 'none' and v != '':
                if v not in info:
                    info[v] = track
                    revinfo[track] = v
                    track += 1
        for i in range(db.shape[0]):
            v = db[i,j].strip().lower()

            if 'none' == v:
                n_v = -1.
            elif '' == v:
                n_v = -1.
            else:
                n_v = info[v]

            f_db[i,j] = n_v
        all_revinfo[j] = revinfo
    with open("./util/info_map.pkl", 'wb') as f:
        pickle.dump(all_revinfo, f)
    for i in range(db.shape[0]):
        for j in range(db.shape[1]):
            v = db[i,j].strip()

            if j not in list(range(12)):
                if 'none' == v:
                    n_v = 0.
                elif '' == v:
                    n_v = 0.
                else:
                    nums = re.findall(r"[-+]?(?:\d*\.*\d+)[%]?", v)
                    if len(nums) > 0:
                        num = nums[0]
                        if "%" == num[-1]:
                            n_v = float(num[:-1])/100.
                        else:
                            n_v = float(num)
                            if (j < 26) and (n_v > 1.):
                                n_v = n_v/100.
                    else:
                        n_v = 0.

                f_db[i,j] = n_v

    # + clean/fill
    cols = []
    remaining_idxs = []
    num_cols = f_db.shape[-1]
    for i in range(num_cols):
        if i >= (num_cols-9):
            cols.append(np.expand_dims(f_db[:,i], 1))
            remaining_idxs.append(i)
            continue
        if 0. == sum(f_db[:,i]) or np.all(f_db[:,i] == -1.):
            continue
        else:
            cols.append(np.expand_dims(f_db[:,i], 1))
            remaining_idxs.append(i)
    f_db = np.concatenate(cols, 1)

    db = f_db

    print("proc (clean, filter) db.shape")
    print(db.shape)

    return db, remaining_idxs

def split_data(db):
    x_db = db[:,:-11]
    y_db = db[:,-11:]
    x_idxs = list(range(len(x_db)))
    np.random.shuffle(x_idxs)
    x_db = x_db[x_idxs]
    y_db = y_db[x_idxs]
    frac = 0.90
    train_x_db = x_db[:int(len(x_db)*frac)].astype(float)
    test_x_db = x_db[int(len(x_db)*frac):].astype(float)
    train_y_db = y_db[:int(len(x_db)*frac)].astype(float)
    test_y_db = y_db[int(len(x_db)*frac):].astype(float)

    return train_x_db, train_y_db, test_x_db, test_y_db

def dset_save(tr_x, tr_y, te_x, te_y):
    np.save("./save/train_data.npy", tr_x)
    np.save("./save/train_labels.npy", tr_y)
    np.save("./save/test_data.npy", te_x)
    np.save("./save/test_labels.npy", te_y)
def dset_load():
    tr_x = np.load("./save/train_data.npy")
    tr_y = np.load("./save/train_labels.npy")
    te_x = np.load("./save/test_data.npy")
    te_y = np.load("./save/test_labels.npy")

    return tr_x, tr_y, te_x, te_y

def subset_data(tr_x, tr_y, te_x, te_y, label_i):
    pred_idx = label_i

    # min/max strengths
    new_x = []
    new_y = []
    for i in range(len(tr_x)):
        d_x = tr_x[i]
        d_y = tr_y[i]
        if d_y[2] > 0.:
            new_x.append(np.concatenate((d_x, [0]),0))
            new_y.append([d_y[2],0.,0.,0.,0.])
        if d_y[0] > 0. and d_y[0] != d_y[2]:
            new_x.append(np.concatenate((d_x, [1]),0))
            new_y.append([d_y[0],0.,0.,0.,0.])
        if d_y[1] > 0. and d_y[1] != d_y[2]:
            new_x.append(np.concatenate((d_x, [2]),0))
            new_y.append([d_y[1],0.,0.,0.,0.])
        if d_y[5] > 0.:
            new_x.append(np.concatenate((d_x, [0]),0))
            new_y.append([0.,d_y[5],0.,0.,0.])
        if d_y[3] > 0. and d_y[3] != d_y[5]:
            new_x.append(np.concatenate((d_x, [1]),0))
            new_y.append([0.,d_y[3],0.,0.,0.])
        if d_y[4] > 0. and d_y[4] != d_y[5]:
            new_x.append(np.concatenate((d_x, [2]),0))
            new_y.append([0.,d_y[4],0.,0.,0.])
        if d_y[8] > 0.:
            new_x.append(np.concatenate((d_x, [0]),0))
            new_y.append([0.,0.,d_y[8],0.,0.])
        if d_y[6] > 0. and d_y[6] != d_y[8]:
            new_x.append(np.concatenate((d_x, [1]),0))
            new_y.append([0.,0.,d_y[6],0.,0.])
        if d_y[7] > 0. and d_y[6] != d_y[8]:
            new_x.append(np.concatenate((d_x, [2]),0))
            new_y.append([0.,0.,d_y[7],0.,0.])
        if d_y[9] > 0.:
            new_x.append(np.concatenate((d_x, [-1]),0))
            new_y.append([0.,0.,0.,d_y[9],0.])
        if d_y[10] > 0.:
            new_x.append(np.concatenate((d_x, [-1]),0))
            new_y.append([0.,0.,0.,0.,d_y[10]])
    tr_x = np.array(new_x)
    tr_y = np.array(new_y)
    new_x = []
    new_y = []
    for i in range(len(te_x)):
        d_x = te_x[i]
        d_y = te_y[i]
        if d_y[2] > 0.:
            new_x.append(np.concatenate((d_x, [0]),0))
            new_y.append([d_y[2],0.,0.,0.,0.])
        if d_y[0] > 0. and d_y[0] != d_y[2]:
            new_x.append(np.concatenate((d_x, [1]),0))
            new_y.append([d_y[0],0.,0.,0.,0.])
        if d_y[1] > 0. and d_y[1] != d_y[2]:
            new_x.append(np.concatenate((d_x, [2]),0))
            new_y.append([d_y[1],0.,0.,0.,0.])
        if d_y[5] > 0.:
            new_x.append(np.concatenate((d_x, [0]),0))
            new_y.append([0.,d_y[5],0.,0.,0.])
        if d_y[3] > 0. and d_y[3] != d_y[5]:
            new_x.append(np.concatenate((d_x, [1]),0))
            new_y.append([0.,d_y[3],0.,0.,0.])
        if d_y[4] > 0. and d_y[4] != d_y[5]:
            new_x.append(np.concatenate((d_x, [2]),0))
            new_y.append([0.,d_y[4],0.,0.,0.])
        if d_y[8] > 0.:
            new_x.append(np.concatenate((d_x, [0]),0))
            new_y.append([0.,0.,d_y[8],0.,0.])
        if d_y[6] > 0. and d_y[6] != d_y[8]:
            new_x.append(np.concatenate((d_x, [1]),0))
            new_y.append([0.,0.,d_y[6],0.,0.])
        if d_y[7] > 0. and d_y[7] != d_y[8]:
            new_x.append(np.concatenate((d_x, [2]),0))
            new_y.append([0.,0.,d_y[7],0.,0.])
        if d_y[9] > 0.:
            new_x.append(np.concatenate((d_x, [-1]),0))
            new_y.append([0.,0.,0.,d_y[9],0.])
        if d_y[10] > 0.:
            new_x.append(np.concatenate((d_x, [-1]),0))
            new_y.append([0.,0.,0.,0.,d_y[10]])
    te_x = np.array(new_x)
    te_y = np.array(new_y)

    # combine flex 1 and flex 2
    if 1 == pred_idx:
        new_x = []
        new_y = []
        for i in range(len(tr_x)):
            y_1 = tr_y[i][1]
            y_2 = tr_y[i][2]
            if y_1 != y_2:
                new_x.append(tr_x[i])
                new_x.append(tr_x[i])
                new_y.append(y_1)
                new_y.append(y_2)
        tr_x = np.array(new_x)
        tr_y = np.array(new_y)
        new_x = []
        new_y = []
        for i in range(len(te_x)):
            y_1 = te_y[i][1]
            y_2 = te_y[i][2]
            if y_1 != y_2:
                new_x.append(te_x[i])
                new_x.append(te_x[i])
                new_y.append(y_1)
                new_y.append(y_2)
        te_x = np.array(new_x)
        te_y = np.array(new_y)

    if pred_idx in [2,3]:
        tpi = pred_idx+1
    else:
        tpi = pred_idx
    idxs = []
    for i,v in enumerate(tr_y):
        if 1 == pred_idx:
            if v > 0.:
                idxs.append(i)
        else:
            if v[tpi] > 0.:
                idxs.append(i)
    tr_x = tr_x[idxs]
    tr_y = tr_y[idxs]
    idxs = []
    for i,v in enumerate(te_y):
        if 1 == pred_idx:
            if v > 0.:
                idxs.append(i)
        else:
            if v[tpi] > 0.:
                idxs.append(i)
    te_x = te_x[idxs]
    te_y = te_y[idxs]

    if pred_idx in [2,3]:
        tr_x = tr_x[:,:-1]
        te_x = te_x[:,:-1]

    # get labels
    if 1 != pred_idx:
        tr_y = tr_y[:,tpi]
        te_y = te_y[:,tpi]

    return tr_x, tr_y, te_x, te_y

def featengr_data(tr_x, tr_y, te_x, te_y, headers, bidxs):
    # remove empty rows
    idxs = []
    for i,d in enumerate(tr_x):
        empty = True
        for j in range(tr_x.shape[-1]):
            v = d[j]
            if j < 12:
                if -1. != v:
                    empty = False
                    break
            else:
                if 0. != v:
                    empty = False
                    break
        if not empty:
            idxs.append(i)
    tr_x = tr_x[idxs]
    tr_y = tr_y[idxs]
    idxs = []
    for i,d in enumerate(te_x):
        empty = True
        for j in range(te_x.shape[-1]):
            v = d[j]
            if j < 12:
                if -1. != v:
                    empty = False
                    break
            else:
                if 0. != v:
                    empty = False
                    break
        if not empty:
            idxs.append(i)
    te_x = te_x[idxs]
    te_y = te_y[idxs]

    if len(bidxs) > 0:
        tr_x = tr_x[:,bidxs]
        te_x = te_x[:,bidxs]

    return tr_x, tr_y, te_x, te_y

def analyze(tr_x, tr_y, te_x, te_y, headers, label_i):
    # pca
    pca = PCA(n_components=2)
    pca.fit(tr_x)

    trans_x = pca.transform(tr_x)

    fig, ax = plt.subplots()
    ax.scatter(trans_x[:,0], trans_x[:,1])
    plt.savefig("./figs/featanal_pca_{}.png".format(label_i), bbox_inches="tight")
    plt.clf()
    plt.close()

    # tsne
    pca = PCA(n_components=6)
    dense_x = pca.fit_transform(tr_x)
    tsne = TSNE(n_components=2)
    trans_x = tsne.fit_transform(dense_x)

    fig, ax = plt.subplots()
    ax.scatter(trans_x[:,0], trans_x[:,1])
    plt.savefig("./figs/featanal_tsne_{}.png".format(label_i), bbox_inches="tight")
    plt.clf()
    plt.close()

    best_idxs = []

    # multicollinearity check
    tr_all = pd.concat([pd.DataFrame.from_records(tr_x), pd.DataFrame.from_records(np.expand_dims(tr_y,1))], axis=1)
    plt.figure(figsize=(10,8))
    corr_mat = tr_all.corr()[0].to_numpy()[:,1].squeeze()
    corr_idxs = []
    for c_i,v in enumerate(corr_mat):
        if v > 0.01:
            corr_idxs.append(c_i)
    sns.heatmap(tr_all.corr(), cmap=sns.cm.rocket_r)#, annot=True)
    plt.savefig("./figs/featanal_multicollin_{}.png".format(label_i), bbox_inches="tight")
    plt.clf()
    plt.close()
    if 1 == featpick:
        best_idxs = corr_idxs
        if corr_mat.shape[0]-1 in best_idxs:
            del(best_idxs[best_idxs.index(corr_mat.shape[0]-1)])

    # F-regression test
    selector = SelectKBest(f_regression, k=1)
    selector.fit(tr_x, tr_y)
    with open("./figs/f-regression_results_{}.txt".format(label_i), 'w') as f:
        for i,pv in enumerate(selector.pvalues_):
            f.write("{}: {}\n".format(headers[i], pv))
    for i,pv in enumerate(selector.pvalues_):
        if 2 == featpick:
            if pv > 0.35:
                best_idxs.append(i)

    return best_idxs

def proc(train_x_db, train_y_db, test_x_db, test_y_db, headers, label_i):
    # training
    y = train_y_db
    X = train_x_db
    regr = make_pipeline(RobustScaler(), RandomForestRegressor())

    # hparam search
    if do_hparam:
        regr = RandomForestRegressor()
        parameters = {'n_estimators': [50,100,200,500,1000], 'max_depth': [None, 3, 5], 'max_features': [None, 0.5, "sqrt"]}
        clf = GridSearchCV(regr, parameters, cv=5)
        clf.fit(X, y)
        df = pd.DataFrame(clf.cv_results_)
        regr = make_pipeline(RobustScaler(), clf.best_estimator_)

    # - learning curve
    max_samps = int(len(X)*0.78)
    samp_arr = [int(max_samps//2), max_samps] # NOTE could be configured
    LearningCurveDisplay.from_estimator(regr, X, y, train_sizes=samp_arr, cv=5)
    plt.savefig("./figs/train_curve_{}.png".format(label_i), bbox_inches="tight")
    plt.clf()
    plt.close()

    # - fit
    regr.fit(X, y)

    joblib.dump(regr, "./save/pred_model_{}.joblib".format(label_i), compress=3)

    return regr

def run():
    db, cc_db = load_data()

    val_idxs = np.concatenate((cat_idxs, mat_idxs, str_idxs), 0)
    headers = db[0][val_idxs]

    db = db[1:]
    db = db[:,val_idxs]

    # data proc stage 1: cleaning of invalid data, dataset-based filtering
    db, r_idxs = clean(db)
    headers = headers[r_idxs]
    with open("./util/header_map.txt", 'w') as f:
        f.write(",".join(headers))

    # attach cost/carbon values
    for i in range(cc_db.shape[0]):
        if 0 == i:
            continue
        for j in range(cc_db.shape[1]):
            if "none" == cc_db[i][j]:
                cc_db[i][j] = 0.
            else:
                cc_db[i][j] = float(cc_db[i][j])
    db = np.concatenate((db, cc_db[1:]), 1)
    headers = np.concatenate((headers, ["Carbon Footprint", "Cost"]), 0)

    if os.path.exists("./save/train_data.npy"):
        tr_x_o, tr_y_o, te_x_o, te_y_o = dset_load()
    else:
        tr_x_o, tr_y_o, te_x_o, te_y_o = split_data(db)
        dset_save(tr_x_o, tr_y_o, te_x_o, te_y_o)

    log_name = "matpred_results_{}.log".format(datetime.today().strftime('%Y-%m-%d'))
    with open("./logs/{}".format(log_name), 'w') as f:
        f.write("MatPred Results Log\n")
        f.write("-------------------\n\n")

    pred_idxs = [0,1,2,3]
    models = []
    preds_all = []
    preds_tr_all = []
    tr_xs = []
    tr_ys = []
    te_xs = []
    te_ys = []
    for pred_idx in pred_idxs:
        with open("./logs/{}".format(log_name), 'a') as f:
            f.write("\nFor pred_idx: {}\n".format(pred_idx))

        # data proc stage 2 (optional): label-based filtering
        tr_x, tr_y, te_x, te_y = subset_data(tr_x_o, tr_y_o, te_x_o, te_y_o, pred_idx)

        best_idxs = []
        if do_analysis:
            best_idxs = analyze(tr_x, tr_y, te_x, te_y, headers, pred_idx)

        # data proc stage 3: feature selection/engineering (from analysis)
        tr_x, tr_y, te_x, te_y = featengr_data(tr_x, tr_y, te_x, te_y, headers, best_idxs)

        seeds = [1,2,3,4,5]
        preds = []
        preds_tr = []
        r2s = []
        rmses = []
        for seed_i,seed in enumerate(seeds):
            print("--- seed {}".format(seed))
            np.random.seed(seed)

            model = proc(tr_x, tr_y, te_x, te_y, headers, pred_idx)

            pred_y = model.predict(te_x)
            preds.append(pred_y)
            pred_tr_y = model.predict(tr_x)
            preds_tr.append(pred_tr_y)

            r2 = model.score(te_x, te_y)
            print("check, seed {}".format(seed))
            print(r2)

            tr_r2 = model.score(tr_x, tr_y)
            te_r2 = model.score(te_x, te_y)
            tr_rmse = root_mean_squared_error(tr_y, pred_tr_y)
            te_rmse = root_mean_squared_error(te_y, pred_y)
            r2s.append(te_r2)
            rmses.append(te_rmse)

            with open("./logs/{}".format(log_name), 'a') as f:
                f.write("{} (seed {}) (train) r2 = {}\n".format(pred_idx, seed, tr_r2))
                f.write("{} (seed {}) (train) rmse = {}\n".format(pred_idx, seed, tr_rmse))
                f.write("{} (seed {}) (test) r2 = {}\n".format(pred_idx, seed, te_r2))
                f.write("{} (seed {}) (test) rmse = {}\n".format(pred_idx, seed, te_rmse))

        models.append(model)
        preds_all.append(preds)
        preds_tr_all.append(preds_tr)
        tr_xs.append(tr_x)
        tr_ys.append(tr_y)
        te_xs.append(te_x)
        te_ys.append(te_y)

        print("overall r2")
        print(np.mean(r2s))
        print(np.std(r2s))
        print("overall rmse")
        print(np.mean(rmses))
        print(np.std(rmses))
        with open("./logs/{}".format(log_name), 'a') as f:
            f.write("{} (overall) (test) r2 = {} +- {}\n".format(pred_idx, np.mean(r2s), np.std(r2s)))
            f.write("{} (overall) (test) rmse = {} +- {}\n".format(pred_idx, np.mean(rmses), np.std(rmses)))

    # figures
    # - r2
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(1,2,figsize=(14,8))
    regr = models[0]
    pred_tr_y = preds_tr_all[0][-1] # for now
    pred_y = preds_all[0][-1]
    tr_y = tr_ys[0]
    te_y = te_ys[0]
    axs[0].scatter(tr_y, pred_tr_y, c="blue", marker=".", label="Train")
    axs[0].scatter(te_y, pred_y, c="red", marker="D", label="Test")
    left_min = min([te_y[np.argmin(te_y)], pred_y[np.argmin(pred_y)], tr_y[np.argmin(tr_y)], pred_tr_y[np.argmin(pred_tr_y)]])-1
    right_max = max([te_y[np.argmax(te_y)], pred_y[np.argmax(pred_y)], tr_y[np.argmax(tr_y)], pred_tr_y[np.argmax(pred_tr_y)]])+1
    axs[0].plot([left_min, right_max], [left_min, right_max], color="black", lw=1.5)
    axs[0].legend()
    axs[0].set_xlabel("Actual Compressive Strength [MPa]")
    axs[0].set_ylabel("Predicted Compressive Strength [MPa]")
    regr = models[1]
    pred_tr_y = preds_tr_all[1][-1]
    pred_y = preds_all[1][-1]
    tr_y = tr_ys[1]
    te_y = te_ys[1]
    axs[1].scatter(tr_y, pred_tr_y, c="blue", marker=".", label="Train")
    axs[1].scatter(te_y, pred_y, c="red", marker="D", label="Test")
    left_min = min([te_y[np.argmin(te_y)], pred_y[np.argmin(pred_y)], tr_y[np.argmin(tr_y)], pred_tr_y[np.argmin(pred_tr_y)]])-1
    right_max = max([te_y[np.argmax(te_y)], pred_y[np.argmax(pred_y)], tr_y[np.argmax(tr_y)], pred_tr_y[np.argmax(pred_tr_y)]])+1
    axs[1].plot([left_min, right_max], [left_min, right_max], color="black", lw=1.5)
    axs[1].legend()
    axs[1].set_xlabel("Actual Flexural Strength [MPa]")
    axs[1].set_ylabel("Predicted Flexural Strength [MPa]")
    plt.savefig("./figs/r2_plot.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    # - box-and-whisker
    regr = models[0]
    preds_y = preds_all[0]
    errors = []
    errors_abs = []
    for pred_y in preds_y:
        error_y = []
        error_abs_y = []
        for p_i,pred in enumerate(pred_y):
            error = pred - te_ys[0][p_i]
            error_y.append(error)
            error_abs_y.append(abs(error))
        errors.append(error_y)
        errors_abs.append(error_abs_y)
    errors_c_mean = np.mean(errors, 0)
    errors_c_std = np.std(errors, 0)
    errors_c_mean_abs = np.mean(errors_abs, 0)
    preds_c_mean = np.mean(preds_y, 0)
    regr = models[1]
    preds_y = preds_all[1]
    errors = []
    errors_abs = []
    for pred_y in preds_y:
        error_y = []
        error_abs_y = []
        for p_i,pred in enumerate(pred_y):
            error = pred - te_ys[1][p_i]
            error_y.append(error)
            error_abs_y.append(abs(error))
        errors.append(error_y)
        errors_abs.append(error_abs_y)
    errors_f_mean = np.mean(errors, 0)
    errors_f_std = np.std(errors, 0)
    errors_f_mean_abs = np.mean(errors_abs, 0)
    preds_f_mean = np.mean(preds_y, 0)
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots()
    labels = ["Compressive Strength [MPa]", "Flexural Strength [MPa]"]
    bp = axs.boxplot([errors_c_mean, errors_f_mean], tick_labels=labels)
    axs.set_xticks(list(range(1,len(labels)+1)), labels, rotation=30, ha='right')
    axs.set_ylabel("Prediction Errors [MPa]")
    plt.savefig("./figs/boxandwhisker_plot.png", bbox_inches="tight")
    plt.clf()
    plt.close()
    print("boxplot medians:")
    for medline in bp['medians']:
        linedata = medline.get_ydata()
        print(linedata[0])

    # - bar
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(1,2,figsize=(14,8))
    c_idxs = np.argsort(errors_c_mean_abs)
    errors_c_mean_abs = errors_c_mean_abs[c_idxs]
    te_y_c = te_ys[0][c_idxs]
    pr_c = preds_c_mean[c_idxs]
    f_idxs = np.argsort(errors_f_mean_abs)
    errors_f_mean_abs = errors_f_mean_abs[f_idxs]
    te_y_f = te_ys[1][f_idxs]
    pr_f = preds_f_mean[f_idxs]
    for i in range(2):
        if 0 == i:
            y_data = te_y_c[:10]
            p_data = pr_c[:10]
            e_data = errors_c_mean_abs[:10]
            l1 = "Actual Compressive Strength"
            l2 = "Predicted Compressive Strength"
        else:
            y_data = te_y_f[:10]
            p_data = pr_f[:10]
            e_data = errors_f_mean_abs[:10]
            l1 = "Actual Flexural Strength"
            l2 = "Predicted Flexural Strength"
        x = np.arange(10)
        width = 0.25
        multiplier = 0
        offset = width*multiplier
        axs[i].bar(x+offset, y_data, width, label=l1)
        multiplier += 1
        offset = width*multiplier
        axs[i].bar(x+offset, p_data, width, label=l2)
        axs[i].errorbar(x+offset, p_data, yerr=e_data, fmt=',', color='r')
        axs[i].set_xticks(list(range(0,10)), list(range(10)))#, rotation=30, ha='right')
        axs[i].set_xlabel("Sample Number")
        axs[i].legend()
        if 0 == i:
            axs[i].set_ylabel("Compressive Strength [MPa]")
        else:
            axs[i].set_ylabel("Flexural Strength [MPa]")
    plt.savefig("./figs/bar_plot.png", bbox_inches="tight")
    plt.clf()
    plt.close()

if __name__ == "__main__":
    run()
