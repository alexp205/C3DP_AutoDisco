import os
from google import genai
from google.genai import types
import pathlib
import csv
import numpy as np
import time
import pandas as pd
import pymupdf
from difflib import SequenceMatcher

API_KEY = "TODO"
data_dirs = ["./db/papers/"]
db_path = "./db/refs/ref_db.npy"
db_new_path = "./db/refs/new_papers.npy"

client = genai.Client(api_key=API_KEY)

with open("./util/starter_auth.txt", 'r') as f:
    approve_auth = f.read()
approve_auth = approve_auth.split('\n')
with open("./util/starter_ven.txt", 'r') as f:
    approve_ven = f.read()
approve_ven = approve_ven.split('\n')
with open("../prompts/rs_1.txt", 'r') as f:
    prompt = f.read()

ref_db = np.array([])
if os.path.exists(db_path):
    print("Loading refs database...")
    ref_db = np.load(db_path)
if 0 == len(ref_db):
    # starter papers
    with open("./util/eval_plist.txt", 'r') as f:
        def_reflist = f.readlines()
    for l in def_reflist:
        l = l.strip().lower()
        if 0 == len(ref_db):
            ref_db = np.array([[l, "1"]])
        else:
            ref_db = np.append(ref_db, np.array([[l, "1"]]), axis=0)
    # seed approved list
    for data_dir in data_dirs:
        for (root, dirs, files) in os.walk(data_dir, topdown=True):
            for f in files:
                f_id = f[:-4]
                print(f_id)
                x_fpath = root + f
                fsize = os.path.getsize(x_fpath)/1e6
                doc = pymupdf.open(x_fpath)
                text = chr(12).join([page.get_text() for page in doc])

                contents = [
                    """
                    Here is the text of a research paper:
                    {}

                    What is the title of this paper? Extract and respond with ONLY the paper's title. Convert any special characters to standard human-interpretable characters. If no title is discernible, simply respond with nothing.
                    """.format(text)
                ]
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                )
                l = response.text.strip().lower()
                ref_db = np.append(ref_db, np.array([[l, "1"]]), axis=0)
np.save(db_path, ref_db)

def extract_l(line):
    l = line

    s_i = 0
    e_i = len(l)
    c = l[s_i]
    while not c.isalnum():
        s_i += 1
        c = l[s_i]
    c = l[e_i-1]
    while e_i > s_i and not c.isalnum():
        e_i -= 1
        c = l[e_i-1]
    l = l[s_i:e_i]

    return l

def run():
    global ref_db
    new_db = []

    for data_dir in data_dirs:
        for (root, dirs, files) in os.walk(data_dir, topdown=True):
            for f in files:
                f_id = f[:-4]
                print()
                print(f_id)
                x_fpath = root + f
                fsize = os.path.getsize(x_fpath)/1e6
                print(fsize)
                if fsize >= 20:
                    print("WARNING - file size is very large (>= 20 MB)")
                doc = pymupdf.open(x_fpath)
                text = chr(12).join([page.get_text() for page in doc])

                contents = [
                    prompt.format(text)
                ]
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                )

                if pd.isnull(response.text):
                    continue
                resp = response.text.strip().split('\n')

                print("ref_db.shape - before")
                print(ref_db.shape)
                for l in resp:
                    if 0 == len(l):
                        continue
                    l = l.strip().lower()

                    #<paper title> | <author list> | <publishing venue>
                    vals = l.split('|')
                    for vi in range(len(vals)):
                        vals[vi] = vals[vi].strip()
                    assert 3 == len(vals)
                    valid = False
                    val_code = "0"
                    if "rilem" in vals[2]:
                        valid = True
                        val_code = "1"
                    if not valid:
                        for app in approve_auth:
                            s = SequenceMatcher(None, app.strip().lower(), vals[1].strip().lower())
                            if (app.strip().lower() in vals[1]) or (s.ratio() > 0.9):
                                valid = True
                                val_code = "1"
                                break
                    if not valid:
                        for app in approve_ven:
                            s = SequenceMatcher(None, app.strip().lower(), vals[2].strip().lower())
                            if (app.strip().lower() in vals[2]) or (s.ratio() > 0.9):
                                valid = True
                                break

                    if valid:
                        l = extract_l(vals[0])
                        if l not in ref_db[:,0]:
                            ref_db = np.append(ref_db, np.array([[l, val_code]]), axis=0)
                            if 0 == len(new_db):
                                new_db = np.array([[l, vals[1], vals[2]]])
                            else:
                                new_db = np.append(new_db, np.array([[l, vals[1], vals[2]]]), axis=0)
                        else:
                            print("DUPLICATE FOUND, NOT ADDING")
                print("ref_db.shape - after")
                print(ref_db.shape)
                print("updated new_db.shape")
                print(new_db.shape)

                print("Saving database...")
                np.save(db_path, ref_db)

                print("sleeping to avoid overload...")
                time.sleep(3)

    print("Saving NEW papers")
    np.save(db_new_path, new_db)

if __name__ == "__main__":
    run()
