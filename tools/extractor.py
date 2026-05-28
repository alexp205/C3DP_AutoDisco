import os
from google import genai
from google.genai import types
import pathlib
import csv
import numpy as np
import pymupdf
import time
import re
import pandas as pd
from difflib import SequenceMatcher

API_KEY = "TODO"
data_dirs = ["./db/papers/"]
db_path = "./db/db.npy"
new_db_path = "./db/db_postproc.npy"
m_name = "gemini-2.5-pro"

header_str = "Paper ID | Paper Title | Fiber 1 (type -or- none) | Fiber 1 %weight | Fiber 1 %volume | Fiber 2 (type -or- none) | Fiber 2 %weight | Fiber 2 %volume | Fiber 3 (type -or- none) | Fiber 3 %weight | Fiber 3 %volume | Binder (type -or- none) | Binder 1 (type -or- none) | Binder 1 %weight | Binder 2 (type -or- none) | Binder 2 %weight | Water %weight | Additive 1 (type -or- none) | Additive 1 %weight | Additive 2 (type -or- none) | Additive 2 %weight | Additive 3 (type -or- none) | Additive 3 %weight | Additive 4 (type -or- none) | Additive 4 %weight | Aggregate (type -or- none) | Aggregate %weight | Sample Form Factor | Total Sample Width (mm) | Total Sample Depth (mm) | Total Sample Height (mm) | Printed Layer Width (mm) | Printed Layer Height (mm) | Print Speed | Layer Interval (i.e. time between layer deposition) | Compression/Flexural Strength Test Standard | Compressive Strength - Min (MPa) | Compressive Strength - Max (MPa) | Compressive Strength - Average (MPa) | LV I Flexural Strength - Min (MPa) | LV I Flexural Strength - Max (MPa) | LV I Flexural Strength - Average (MPa) | LV II Flexural Strength - Min (MPa) | LV II Flexural Strength - Max (MPa) | LV II Flexural Strength - Average (MPa) | Paper Extraction Location(s) | External Reference (Optional) | Flag - Sample | Flag - Printing | Flag - Loading | Flag - Strength | Warnings"
temp = header_str.split('|')
print("len(headers)")
print(len(temp))
db = np.array([])
if os.path.exists(db_path):
    print("Loading database...")
    db = np.load(db_path)

client = genai.Client(api_key=API_KEY)

def text_convert(resp, f_id=None, mode=1):
    reader = csv.reader(resp.splitlines(), delimiter="|")
    temp = []
    for r in reader:
        temp.append(r)
    if 0 == mode:
        conv_data = np.array(temp)
    else:
        temp = np.array(temp)
        id_col = np.full((len(temp),1), f_id)
        conv_data = np.concatenate((id_col, temp), 1)

    return conv_data

def dup_check(dbref, d, marked={}):
    dup = True

    if len(marked) > 0:
        if 0 == len(dbref):
            return False
    else:
        if 1 == len(dbref):
            return False
    db_ref = np.concatenate((dbref[1:,2:35], dbref[1:,35:44]), 1)
    d_ref = np.concatenate((d[2:35], d[35:44]), 0)
    for i_ref,ref in enumerate(db_ref):
        if i_ref in marked:
            continue
        for ei in range(len(d_ref)):
            if d_ref[ei].strip().lower() == ref[ei].strip().lower() \
               or d_ref[ei].strip().lower() in ref[ei].strip().lower() \
               or ref[ei].strip().lower() in d_ref[ei].strip().lower():
                continue
            nums_dref = re.findall(r"[-+]?(?:\d*\.*\d+)", d_ref[ei])
            nums_ref = re.findall(r"[-+]?(?:\d*\.*\d+)", ref[ei])
            if len(nums_dref) != len(nums_ref):
                dup = False
                break
            if len(nums_dref) > 0:
                for ndr in nums_dref:
                    if ndr not in nums_ref:
                        dup = False
                        break
            if not dup:
                break
        if not dup:
            break

    return dup

def loadsamp_proc(resp):
    load_desc = ""
    samp_desc = ""
    if "Loading Description:" in resp:
        temp = resp.split("Loading Description:")[-1].strip()
        if "Sample Reference List:" in resp:
            temp = temp.split("Sample Reference List:")[0].strip()
            load_desc = "Loading Description: " + temp
        else:
            load_desc = "Loading Description: " + temp
    if "Sample Reference List:" in resp:
        temp = resp.split("Sample Reference List:")[-1].strip()
        samp_desc = "Sample Reference List: " + temp

    return load_desc, samp_desc
def load_proc(resp):
    load_desc = ""
    if "Loading Description:" in resp:
        temp = resp.split("Loading Description:")[-1].strip()
        load_desc = "Loading Description: " + temp

    return load_desc

def proc_paper(doc, t_xpath, f_id, size):
    # get loading direction of paper wrt our loading direction
    with open("./util/matlist.txt", 'r') as f:
        matlist = f.read()
    # - text
    text = chr(12).join([page.get_text() for page in doc])
    with open("../prompts/p1.txt", 'r') as f:
        p = f.read()
    contents = [
        p.format(matlist, text)
    ]
    if len(p.format(matlist, text))//4 > 200000:
        print("!!! WARNING: long token count {} !!!".format(len(p.format(matlist, text))//4))
    trying = True
    while trying:
        try:
            response = client.models.generate_content(
                model=m_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.01,
                    topP=0.95,
                    seed=1
                )
            )
            trying = False
        except genai.errors.ClientError:
            print("resource error, sleeping for a while...")
            time.sleep(60)
    load_info, samp_info = loadsamp_proc(response.text)

    # extract from text and tables
    exds = [[], []]
    pps = ["../prompts/p2a.txt", "../prompts/p2b.txt"]
    l_ipps = ["../prompts/p3a.txt", "../prompts/p3b.txt"]
    l_ipps_2 = ["../prompts/p4a.txt", "../prompts/p4b.txt"]
    for i_p,pp in enumerate(pps):
        with open(pp, 'r') as f:
            p = f.read()
        contents = [
            p.format(matlist, load_info, samp_info, text)
        ]
        if len(p.format(matlist, load_info, samp_info, text))//4 > 200000:
            print("!!! WARNING: long token count {} !!!".format(len(p.format(matlist, load_info, samp_info, text))//4))
        trying = True
        while trying:
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.01,
                        topP=0.95,
                        seed=1
                    )
                )
                trying = False
            except genai.errors.ClientError:
                print("resource error, sleeping for a while...")
                time.sleep(60)
        exd_text = ""
        if response.text:
            exd_text = response.text

        # extract from figures
        store = [exd_text]
        fig_store = []
        for page_i in range(len(doc)):
            page = doc[page_i]
            img_l = page.get_images()
            if not img_l:
                print("no images found on page, continuing...")
            for img_i, img in enumerate(img_l, start=1):
                xref = img[0]
                pix = pymupdf.Pixmap(doc, xref)
                try:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                except ValueError:
                    pix = None
                    continue
                pix.save("cand_fig.png")
                with open("cand_fig.png", 'rb') as f:
                    pic = f.read()

                with open(l_ipps[i_p], 'r') as f:
                    p = f.read()
                contents = [
                    types.Part.from_bytes(
                        data=pic,
                        mime_type='image/png',
                    ),
                    p.format()
                ]
                if len(p.format(matlist, load_info, samp_info, "\n".join(store)))//4 > 200000:
                    print("!!! WARNING: long token count {} !!!".format(len(p.format(matlist, load_info, samp_info, "\n".join(store)))//4))
                trying = True
                while trying:
                    try:
                        response = client.models.generate_content(
                            model=m_name,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                temperature=0.01,
                                topP=0.95,
                                seed=1
                            )
                        )
                        trying = False
                    except genai.errors.ClientError:
                        print("resource error, sleeping for a while...")
                        time.sleep(60)
                if not response.text:
                    pix = None
                    continue
                if not response.text.lower().strip().endswith("none"):
                    with open(l_ipps_2[i_p], 'r') as f:
                        p = f.read()
                    contents = [
                        types.Part.from_bytes(
                            data=pic,
                            mime_type='image/png',
                        ),
                        p.format(matlist, load_info, samp_info, text)
                    ]
                    if len(p.format(matlist, load_info, samp_info, text))//4 > 200000:
                        print("!!! WARNING: long token count {} !!!".format(len(p.format(matlist, load_info, samp_info, text))//4))
                    trying = True
                    while trying:
                        try:
                            response = client.models.generate_content(
                                model=m_name,
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    temperature=0.01,
                                    topP=0.95,
                                    seed=1
                                )
                            )
                            trying = False
                        except genai.errors.ClientError:
                            print("resource error, sleeping for a while...")
                            time.sleep(60)
                    if response.text:
                        store.append(response.text)
                        fig_store.append(response.text)
                pix = None
        exd_fig = "\n".join(fig_store)
        exds[0].append(exd_text)
        exds[1].append(exd_fig)

    # expand strength sample information
    # - check dataset
    for e_i in range(len(exds)):
        exd = exds[e_i]
        for e_ii in range(len(exd)):
            lines = exd[e_ii].split('\n')
            temp = []
            for line in lines:
                temp.append(line.strip().strip('|').strip())
            exd[e_ii] = "\n".join(temp)
    valid = []
    if exds[0][0]:
        temp = text_convert(exds[0][0], f_id)
        if 13 != temp.shape[-1]:
            print("data extraction error --- shape")
            # error out and skip
            return None, True
        temp = np.concatenate((temp[:,:6], np.full((len(temp),6), "none"), temp[:,6:]), 1)
        valid.append(temp)
    else:
        print("no data")
    if exds[0][1]:
        temp = text_convert(exds[0][1], f_id)
        if 16 != temp.shape[-1]:
            print("data extraction error --- shape")
            # error out and skip
            return None, True
        temp = np.concatenate((temp[:,:3], np.full((len(temp),3), "none"), temp[:,3:]), 1)
        valid.append(temp)
    else:
        print("no data")
    if exds[1][0]:
        temp = text_convert(exds[1][0], f_id)
        if 13 != temp.shape[-1]:
            print("data extraction error --- shape")
            # error out and skip
            return None, True
        temp = np.concatenate((temp[:,:6], np.full((len(temp),6), "none"), temp[:,6:]), 1)
        valid.append(temp)
    else:
        print("no data")
    if exds[1][1]:
        temp = text_convert(exds[1][1], f_id)
        if 15 == temp.shape[-1] and temp[-1][-1].strip()[-1].isnumeric():
            temp = np.concatenate((temp, np.full((len(temp),1), "none")), 1)
        if 16 != temp.shape[-1]:
            print("data extraction error --- shape")
            # error out and skip
            return None, True
        temp = np.concatenate((temp[:,:3], np.full((len(temp),3), "none"), temp[:,3:]), 1)
        valid.append(temp)
    else:
        print("no data")
    if not valid:
        print("no data found, continuing")
        return None, True
    gen_db = np.concatenate(valid, 0)
    idxs = []
    for i_d,d in enumerate(gen_db):
        if not ("paper title" in d[1].lower().strip() and "warnings" in d[-1].lower().strip()):
            idxs.append(i_d)
    gen_db = gen_db[idxs]
    marked = {}
    for i_d,d in enumerate(gen_db):
        dup = dup_check(gen_db[:i_d], d, marked)
        if dup:
            print("---duplicate found, skipping")
            marked[i_d] = 1
    idxs = []
    for i_d,d in enumerate(gen_db):
        if i_d not in marked:
            idxs.append(i_d)
    gen_db = gen_db[idxs]

    # - per-sample, material expand parts 1 and 2, print info, form factor and curing info
    pps = ["../prompts/p5.txt", "../prompts/p6.txt", "../prompts/p7.txt"]
    store = []
    for i_p,pp in enumerate(pps):
        partial = []
        for i_d,d in enumerate(gen_db):
            with open(pp, 'r') as f:
                p = f.read()
            sample = d[2]
            if 0 == i_p:
                p = p.format(samp_info, text, sample)
            elif 1 == i_p:
                p = p.format(store[-1], samp_info, text, sample)
            elif 2 == i_p:
                p = p.format(samp_info, text, sample)
            contents = [
                p
            ]
            if len(p)//4 > 200000:
                print("!!! WARNING: long token count {} !!!".format(len(p)//4))
            trying = True
            while trying:
                try:
                    response = client.models.generate_content(
                        model=m_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.01,
                            topP=0.95,
                            seed=1
                        )
                    )
                    trying = False
                except genai.errors.ClientError:
                    print("resource error, sleeping for a while...")
                    time.sleep(60)
            exp = ""
            if response.text:
                exp = response.text
            if exp:
                partial.append(exp)
                temp = text_convert(exp, mode=1)
                if 0 == i_p:
                    cols = 15
                elif 1 == i_p:
                    cols = 10
                elif 2 == i_p:
                    cols = 9
                if cols != temp.shape[-1]:
                    print("data extraction error --- shape")
                    # error out and skip
                    return None, True
        if partial:
            store.append(partial)
        else:
            store.append("")
    upd_db = []
    for j in range(len(store[0])):
        extra = "|".join([store[0][j], store[1][j], store[2][j]])
        extra = text_convert(extra, mode=1)
        upd_db.append(np.concatenate((gen_db[j][:2], extra[0], gen_db[j][3:]), 0))
    gen_db = np.array(upd_db)

    return gen_db, False

def proc_samp(doc, t_xpath, f_id, size, data):
    text = chr(12).join([page.get_text() for page in doc])

    # extract actual perc corrections
    with open("../prompts/sup_p1.txt", 'r') as f:
        p = f.read()
    contents = [
        p.format(text, " | ".join(db[0][2:27]), " | ".join(data[2:27]), densities)
    ]
    if len(p.format(text, " | ".join(db[0][2:27]), " | ".join(data[2:27]), densities))//4 > 200000:
        print("!!! WARNING: long token count {} !!!".format(len(p.format(text, " | ".join(db[0][2:27]), " | ".join(data[2:27]), densities))//4))
    trying = True
    while trying:
        try:
            response = client.models.generate_content(
                model=m_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.01,
                    topP=0.95,
                    seed=1
                )
            )
            trying = False
        except genai.errors.ClientError:
            print("weird af resource error, sleeping for a while...")
            time.sleep(60)
    exd_text = ""
    if response.text:
        exd_text = response.text
    if 0 == len(exd_text):
        print("no data found, continuing")
        return None

    return exd_text

def run():
    global db
    for data_dir in data_dirs:
        ctr = 0
        for (root, dirs, files) in os.walk(data_dir, topdown=True):
            for f in files:
                if ".txt" == f[-4:]:
                    continue

                ctr += 1
        
                f_id = f[:-4]
                print(f_id)

                if len(db) > 0:
                    if f_id in db[:,0]:
                        print("file {} already processed, continuing...".format(f_id))
                        continue

                x_fpath = root + f
                print(os.path.getsize(x_fpath)/1e6) # ~MB
                fsize = os.path.getsize(x_fpath)/1e6
                if fsize >= 20:
                    print("WARNING - file size is very large (>= 20 MB)")
                doc = pymupdf.open(x_fpath)
                t_xpath = pathlib.Path(x_fpath)

                pdata, failed = proc_paper(doc, t_xpath, f_id, fsize)
                if failed:
                    print("error in processing paper {}, SKIPPING".format(f_id))
                    continue

                if 0 == len(db):
                    header = header_str.split("|")
                    for i in range(len(header)):
                        header[i] = header[i].strip()
                    db = np.array([header])
                print("db - before update")
                print(db.shape)
                for d in pdata:
                    dup = dup_check(db, d)
                    if dup:
                        print("---duplicate found, skipping")
                        continue
                    db = np.append(db, np.expand_dims(d,0), axis=0)
                print("db - after update")
                print(db.shape)
        
                print("Saving database...")
                np.save(db_path, db)
    
                print("sleeping to avoid overload...")
                time.sleep(3)

    # perform corrections
    outdata_path = "./db/perc_corrections.npy"
    with open("./util/mat_densities.csv", 'r') as f:
        densities = f.read()
    new_outdata = np.array([])
    header_str = "Original Dataset Index | Paper ID | Material-to-Clarify | Material | Weight%"
    header = header_str.split("|")
    for i in range(len(header)):
        header[i] = header[i].strip()
    new_outdata = np.array([header])
    seen = {}
    count = 0
    for i,d in enumerate(db):
        if 0 == i:
            continue
        # get paper id
        pid = d[0]
        print(pid)
        # retrieve doc
        found = False
        doc = None
        doc_root = None
        for data_dir in possible_data_dirs:
            for (root, dirs, files) in os.walk(data_dir, topdown=True):
                for f in files:
                    if ".txt" == f[-4:]:
                        continue
                    if pid == f[:-4]:
                        print("found {} in {}".format(pid, data_dir))
                        doc = f
                        doc_root = root
                        found = True
                        break
                if found:
                    break
            if found:
                break
        x_fpath = doc_root + doc
        print(os.path.getsize(x_fpath)/1e6) # ~MB
        fsize = os.path.getsize(x_fpath)/1e6
        if fsize >= 20:
            print("WARNING - file size is very large (>= 20 MB)")
        doc = pymupdf.open(x_fpath)
        t_xpath = pathlib.Path(x_fpath)

        correction = proc_samp(doc, t_xpath, pid, fsize, d)

        if correction and "new percentages:" in correction.lower():
            cand_data = correction.lower().split("new percentages:")[-1]
            cand_data = cand_data.strip()
            cands = cand_data.split("\n")
            for cand in cands:
                cand = cand.strip().strip("[]")
                vs = []
                cand_vs = cand.split(",")
                for cand_v in cand_vs:
                    cand_v = cand_v.strip().strip("<>")
                    vs.append(cand_v.strip())
                if 3 != len(vs):
                    continue
                new_data_d = np.array([i, pid, vs[0], vs[1], vs[2]])
                new_outdata = np.append(new_outdata, np.expand_dims(new_data_d,0), axis=0)

        print("Saving corrected samples...")
        np.save(outdata_path, new_outdata)

    # post-process
    mat_idxs = [2, 5, 8, 11, 12, 14, 17, 19, 21, 23, 25]
    str_idxs = [37,38,39,40,41,42,43,44,45]

    # - remove no-strength
    temp_db = [db[0]]
    for d in db[1:]:
        strs = d[str_idxs]
        if all('none' == s for s in strs):
            continue
        temp_db.append(d)
    db = np.array(temp_db)

    # - filter invalid
    idxs = []
    counts = [0 for _ in range(7)]
    for i in range(len(db)):
        if 0 == i:
            idxs.append(i)
            continue
    
        d = db[i]
        warn_v = d[-1].lower()
        flag_samp_v = d[-5].lower()
        flag_print_v = d[-4].lower()
        flag_load_v = d[-3].lower()
        flag_str_v = d[-2].lower()
        form_v = d[28].lower()
        dim_l_v = d[29].lower()
        dim_w_v = d[30].lower()
        dim_h_v = d[31].lower()
        if "none" == dim_l_v:
            dim_l_v = "-1."
        if "none" == dim_w_v:
            dim_w_v = "-1."
        if "none" == dim_h_v:
            dim_h_v = "-1."
        try:
            dim_l_v = float(dim_l_v)
        except:
            dim_l_v = -1.
        try:
            dim_w_v = float(dim_w_v)
        except:
            dim_w_v = -1.
        try:
            dim_h_v = float(dim_h_v)
        except:
            dim_h_v = -1.
        aggsize_v = d[27].lower()
        if "none" == aggsize_v:
            aggsize_v = "0."
        if "-" in aggsize_v:
            aggsize_v = aggsize_v.split("-")[-1]
        try:
            aggsize_v = float(aggsize_v)
        except:
            aggsize_v = 0.
        agg_v = d[25].lower()
        comp_v = d[39].lower()
        flex1_v = d[42].lower()
        flex2_v = d[45].lower()

        valid = True

        # check: 28 day
        if "none" != warn_v and float(flag_samp_v) > 0.:
            if "not 28" in warn_v:
                valid = False
                counts[0] += 1
                continue
        # check: 3d print
        if "none" != warn_v and float(flag_print_v) > 0.:
            if "cast" in warn_v or "not 3d" in warn_v:
                valid = False
                counts[1] += 1
                continue
        # check: comp -> cube/cylinder, flex -> beam/prism
        if "none" != comp_v:
            if not (((dim_l_v == dim_w_v) and (dim_l_v == dim_h_v) and (dim_w_v == dim_h_v)) or "cub" in form_v or "cylind" in form_v or "none" == form_v):
                valid = False
                counts[2] += 1
                continue
        if "none" != flex1_v:
            if not (dim_l_v == dim_w_v):
                valid = False
                counts[2] += 1
                continue
        elif "none" != flex2_v:
            if not (dim_l_v == dim_w_v):
                valid = False
                counts[2] += 1
                continue
        # check: form
        if "none" != form_v:
            if "wall" in form_v:
                valid = False
                counts[3] += 1
                continue
        # check: dims
        if "cub" in form_v or (((dim_l_v == dim_w_v) and (dim_l_v == dim_h_v) and (dim_w_v == dim_h_v)) and not ("cylind" in form_v)):
            if not (-1. == dim_l_v) and not (150 == dim_l_v) and not ((dim_l_v >= 25) and (dim_l_v <= 100)):
                valid = False
                counts[4] += 1
                continue
        elif "cylind" in form_v:
            ratio = dim_h_v/dim_l_v
            if not (-1. == dim_l_v) and ((dim_l_v < 50) or (dim_l_v > 150)) or ((dim_h_v < 100) or (dim_h_v > 300)) or ((ratio < 1.75) or (ratio > 2.25)):
                valid = False
                counts[4] += 1
                continue
        else:
            ratio1 = dim_h_v/dim_l_v
            ratio2 = dim_w_v/dim_l_v
            if not (-1. == dim_l_v) and not (((dim_l_v >= 25) and (dim_l_v <= 150)) or ((dim_w_v >= 25) and (dim_w_v <= 150))) or ((dim_h_v < 87.5) or (dim_h_v > 600)) or not (((ratio1 >= 3.5) and (ratio1 <= 4.25)) or ((ratio2 >= 3.5) and (ratio2 <= 4.25))):
                valid = False
                counts[4] += 1
                continue
        # check: agg size
        if "none" != aggsize_v:
            if aggsize_v > 10:
                valid = False
                counts[5] += 1
                continue
        # check: steel/rebar reinforce
        for ci in mat_idxs:
            if "rebar" in d[ci].lower().strip():
                valid = False
                counts[6] += 1
                continue

        if valid:
            idxs.append(i)
    db = db[idxs]

    header = db[0]
    header_lower = np.array([h.lower() for h in header])
    
    # - weight% value corrections
    outdata_path = "./db/perc_corrections.npy"
    new_outdata = np.load(outdata_path)
    for nod_i,nod in enumerate(new_outdata):
        if 0 == nod_i:
            continue
        db_i = int(nod[0])
        if "%weight" not in nod[2]:
            if "volume" in nod[2]:
                continue
            elif "type" in nod[2]:
                nod[2] = nod[2].split(" (type -or-")[0] + " %weight"
        if nod[2].lower().strip() not in header_lower:
            continue
        db_j = tuple(np.argwhere(header_lower==nod[2].lower().strip())[0])[0].astype(int)
    
        new_mat = nod[-2]
        new_val = nod[-1]
        nums_ref = re.findall(r"[-+]?(?:\d*\.*\d+)[%]?", new_val)
        if 0 == len(nums_ref):
            new_val = -1.
        else:
            nums_ref = nums_ref[-1]
            if "%" == nums_ref[-1]:
                new_val = float(nums_ref[:-1])/100
                if new_val < -1:
                    new_val = float(nums_ref[1:-1])/100
            else:
                new_val = float(nums_ref)
                if new_val < -1:
                    new_val = float(nums_ref[1:])
        if -1. == new_val:
            continue

        db[db_i][db_j] = new_val
        if "water" in nod[2].lower():
            continue
        if "none" == db[db_i][db_j-1].lower().strip() or "not specified" in db[db_i][db_j-1].lower().strip():
            db[db_i][db_j-1] = new_mat
    
    # - material corrections
    with open("./util/matlist.txt", 'r') as f:
        valid_mat_list = f.read()
    valid_mat_list = valid_mat_list.replace("\n", ",").split(",")[:-1]
    df = pd.read_csv("./util/matmap.csv", header=None).to_numpy()
    matmap = {}
    for mc in df:
        orig_mat = ",".join(mc[0].split(",")[:-1]).strip().lower()
        if pd.isnull(mc[1]):
            mapped_mat = ""
        else:
            mapped_mat = mc[1].strip().lower()
        matmap[orig_mat] = mapped_mat
    check_these = []
    bannable_mats = ["graphene fiber", "sulfur", "foaming agent", "encapsulated parafin wax"]
    idxs = []
    for i in range(len(db)):
        if 0 == i:
            idxs.append(i)
            continue
    
        d = db[i]

        keep = True
        for ij,j in enumerate(mat_idxs):
            mat = d[j].strip().lower()
            for banmat in bannable_mats:
                if banmat in mat:
                    keep = False
                    break
            if not keep:
                break
            if "none" == mat:
                continue
            mat = mat.replace("fibre", "fiber")
            mat = mat.replace("plasticiser", "plasticizer")
            mat = mat.replace("stabiliser", "stabilizer")
            found = False
            orig_mat = mat
            if "supplementary cementitious" in mat or "scm" in mat:
                keep = False
                found = True
                break
            if "activator solution" in mat:
                keep = False
                found = True
                break
            if mat in matmap:
                mat = matmap[mat]
                if mat != "":
                    found = True
            if mat in valid_mat_list:
                found = True
            if not found:
                if ij in [0, 1, 2]:
                    if "steel" in mat:
                        mat = "steel fiber"
                        found = True
                    elif "polyvinyl alcohol" in mat or "pva" in mat:
                        mat = "pva fiber"
                        found = True
                    elif "pp" in mat or "poly propylene" in mat or "polypropylene" in mat:
                        mat = "pp fiber"
                        found = True
                    elif "pe" == mat or "polyethylene" in mat or "pe fiber" in mat:
                        mat = "pe fiber"
                        found = True
                    elif "glass fiber" in mat:
                        mat = "glass fiber"
                        found = True
                elif ij in [3, 4, 5]:
                    if "high early strength" in mat or "rapid hardening cement" in mat:
                        mat = "type 3 portland cement"
                        found = True
                    elif not ("type 2" in mat) and ("cementitious" in mat or "portland cement" in mat or "p.o." in mat or "po " in mat or "cement" == mat or "opc" in mat):
                        mat = "type 1 portland cement"
                        found = True
                    elif "type 2" in mat and "cement" in mat:
                        mat = "type 2 portland cement"
                        found = True
                    elif "clay" in mat:
                        mat = "common clay"
                        found = True
                    elif ij in [4, 5]:
                        if "slag" in mat:
                            mat = "slag"
                            found = True
                        elif "fly" in mat and "ash" in mat:
                            mat = "fly ash"
                            found = True
                        elif "brick" in mat:
                            mat = "common clay"
                            found = True
                elif ij in [6, 7, 8, 9]:
                    if "nano-clay" in mat or "nanoclay" in mat:
                        mat = "nano clay"
                        found = True
                    elif "cmc" in mat:
                        mat = "cmc"
                        found = True
                    elif "hpmc" in mat or "hydroxypropyl methylcellulose" in mat or "cellulose ether" in mat:
                        mat = "hpmc"
                        found = True
                    elif "naoh" in mat or ("sodium" in mat and "hydroxide" in mat):
                        mat = "sodium hydroxide"
                        found = True
                    elif "plasticizer" in mat or "plasticiser" in mat:
                        #mat = "plasticizer"
                        mat = "wra"
                        found = True
                    elif "water" in mat and "reduc" in mat:
                        mat = "retarder"
                        found = True
                    elif "rubber" in mat:
                        mat = "crumb rubber"
                        found = True
                    elif "sodium" in mat and "silicate" in mat:
                        mat = "sodium silicate"
                        found = True
                    elif ("calcium" in mat and "silicate" in mat) or "wollastonite" in mat:
                        mat = "calcium silicate"
                        found = True
                    elif "retarder" in mat:
                        mat = "retarder"
                        found = True
                    elif "accelerator" in mat:
                        mat = "accelerator"
                        found = True
                    elif ("viscosity" in mat and "modifying" in mat) or "carboxymethyl" in mat or "stabilizer" in mat:
                        mat = "vma"
                        found = True
                else:
                    if "sand" in mat:
                        mat = "sand"
                        found = True
                    elif ("fine" in mat and "aggregate" in mat) or "fine-" in mat:
                        mat = "sand"
                        found = True
                    elif "brick" in mat:
                        mat = "common clay"
                        found = True
                    elif "perlite" in mat:
                        mat = "perlite"
                if "high early strength" in mat:
                    mat = "type 3 portland cement"
                    found = True
                if "magnesium chloride hexahydrate" in mat:
                    mat = "magnesium chloride"
                    found = True
                if "limestone poder" in mat:
                    mat = "limestone"
                    found = True
                if "saponified" in mat:
                    mat = "surfactant"
                    found = True
                if "micro" in mat:
                    mat = "silica fume"
                    found = True
                if "kaolin" in mat:
                    mat = "kaolin"
                    found = True
                if "pce" in mat:
                    mat = "plasticizer"
                    found = True
                if "cement" in mat:
                    mat = "type 1 portland cement"
                    found = True
                if "fume" in mat:
                    mat = "silica fume"
                    found = True
                if "ash" in mat:
                    mat = "fly ash"
                    found = True
            if not found:
                mat = orig_mat
                match_i = -1
                best_dist = 0.
                for k in range(len(valid_mat_list)):
                    cmat = valid_mat_list[k]
                    s = SequenceMatcher(None, mat.strip().lower(), cmat.strip().lower())
                    cand_dist = s.ratio()
                    if cand_dist > best_dist and cand_dist > 0.7:
                        best_dist = cand_dist
                        match_i = k
                if best_dist > 0.:
                    mat = valid_mat_list[match_i].strip().lower()
                    found = True
                else:
                    check_these.append([i, j])

            if not found:
                keep = False

            if "(potassium silicate powder" in mat: 
                keep = False
                break
            if "coarse" in mat:
                keep = False
                break

            db[i][j] = mat

        if keep:
            idxs.append(i)
    db = db[idxs]

    # - duplicate removal
    mat_idxs_ordered = [25, 2, 5, 8, 12, 14, 17, 19, 21, 23]
    dups_found = 0
    for i in range(len(db)):
        if 0 == i:
            continue
        d = db[i]
        for j in mat_idxs_ordered:
            mat = d[j].strip().lower()
            if "none" == mat:
                continue
    
            for k in mat_idxs_ordered:
                if j == k:
                    continue
                cand = d[k].strip().lower()
                if mat == cand:
                    dups_found += 1
                    db[i][k] = "none"
                    db[i][k+1] = "none"

    # - perc fixes
    perc_idxs = [3,4,6,7,9,10,13,15,16,18,20,22,24,26]
    for i in range(len(db)):
        if 0 == i:
            continue
        perc_is = []
        perc_vals = []
        for pi in perc_idxs:
            val = db[i][pi]
    
            if "none" == val.strip().lower():
                continue
            if "not specified" in val.strip().lower():
                db[i][pi] = "none"
                continue
    
            try:
                new_val = float(val)
            except:
                temp = []
                for v in val.split(" "):
                    if "m3" not in v:
                        temp.append(v)
                val = " ".join(temp)
    
                nums = re.findall(r"[-+]?(?:\d*\.*\d+)[%]?", val)
                if len(nums) > 1:
                    if "100" in nums[-1]:
                        nums = nums[:-1]
                num = nums[-1]
                if "-" == num[0]:
                    num = num[1:]
                if "%" == num[-1]:
                    new_val = float(num[:-1])#/100
                else:
                    new_val = float(num)
    
            perc_is.append(pi)
            perc_vals.append(new_val)
        need_div = not np.all(np.array(perc_vals) <= 1.)
        for j,pi in enumerate(perc_is):
            if need_div:
                db[i][pi] = perc_vals[j]/100.
            else:
                db[i][pi] = perc_vals[j]
    
    # - binder rearranging
    acceptable_binders = ["common clay", "ash", "gypsum", "lime", "illite", "goethite", "dolomite", "kaolin", "slag", "soil", "cement", "weber", "bentonite", "mgo", "uhpc", "magnesium"]
    for i in range(len(db)):
        if 0 == i:
            continue
        d = db[i]
        binds = []
        adds = []
        for j in [12, 14, 17, 19, 21, 23]:
            mat = d[j].strip().lower()
            val = d[j+1].strip().lower()
            if "none" == mat:
                continue
            found = False
            for ab in acceptable_binders:
                if ab in mat:
                    found = True
                    break
            if found:
                binds.append((j, mat, val))
            else:
                adds.append((j, mat, val))
        if len(binds) > 0:
            binds = sorted(binds, key=lambda t: t[2])
        if len(adds) > 0:
            adds = sorted(adds, key=lambda t: t[2])
        all_idxs = [12, 14, 17, 19, 21, 23]
        j = 0
        while j < len(binds) and j < 2:
            v = binds[j]
            db[i][all_idxs[j]] = v[1]
            db[i][all_idxs[j]+1] = v[2]
            j += 1
        while j < 2:
            db[i][all_idxs[j]] = "none"
            db[i][all_idxs[j]+1] = "none"
            j += 1
        k = 0
        while k < len(adds) and k < 4:
            v = adds[k]
            db[i][all_idxs[j+k]] = v[1]
            db[i][all_idxs[j+k]+1] = v[2]
            k += 1
        while j+k < len(all_idxs):
            db[i][all_idxs[j+k]] = "none"
            db[i][all_idxs[j+k]+1] = "none"
            k += 1

    print("saving postprocessed dataset")
    np.save(new_db_path, db)

if __name__ == "__main__":
    run()
