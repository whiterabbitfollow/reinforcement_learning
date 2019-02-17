#!/usr/bin/python
import os 

def get_run_nr(file_dir,starts_with="run"):

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    run_nr = 0

    for d in os.listdir(file_dir):
        if d.startswith(starts_with+"_"):
            run_nr += 1
    return run_nr

