import numpy as np
import os, io
from os.path import join, isdir
from io import BytesIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import boto3
import pandas as pd
import json
import sys


sys.path.append(os.path.abspath('../../'))
from config import *

def qp_eval(x, Q, b):
    return 0.5 * x @ Q @ x + b @ x


def save_data(dimension, resolution, tol):
    if dimension == 5:
        benchmark_name = f"QP-{dimension}d"
        num_instances = 10
        methods = [f"tnc_post_advantage6_sim_qhd_rez{resolution}_T1000",
                   f"tnc_post_advantage6_qhd_rez{resolution}",
                   f"tnc_post_advantage6_qaa_rez{resolution}",
                   f"tnc_post_advantage6_sim_qaa_rez{resolution}_T1000",
                   "tnc", "snopt", "matlab_sqp", "qcqp", "ipopt"]
        methods_short = ["Sim-QHD","DW-QHD","DW-QAA","Sim-QAA","TNC","SNOPT","MATLAB","QCQP","IPOPT"]

    else:
        benchmark_name = f"QP-{dimension}d-5s"
        num_instances = 50
        methods = [f"tnc_post_advantage6_qhd_rez{resolution}", "snopt",
                   f"tnc_post_advantage6_qaa_rez{resolution}", "ipopt", "tnc", "matlab_sqp", "qcqp",]
        methods_short = ["DW-QHD","SNOPT","DW-QAA","IPOPT","TNC","MATLAB","QCQP"]
    print(benchmark_name)

    num_methods = len(methods)
    success_prob = np.zeros((num_methods, num_instances))
    physical_runtime = np.zeros((num_methods, num_instances))
    time_to_solution = np.zeros((num_methods, num_instances))

    for instance in range(num_instances):

        instance_dir = join(DATA_DIR_QP, benchmark_name, f"instance_{instance}")

        # Load instance data
        instance_filename = join(instance_dir, f"instance_{instance}.npy")
        with open(instance_filename, 'rb') as f:
            Q = np.load(f)
            b = np.load(f)
            Q_c = np.load(f)
            b_c = np.load(f)

        # Load Gurobi data (ground truth)
        gurobi_filename = join(instance_dir, f"gurobi_solution_{instance}.npy")
        with open(gurobi_filename, 'rb') as f:
            x_best = np.load(f)
        f_best = qp_eval(x_best, Q, b)


        for idx in range(num_methods):

            method = methods[idx]
 
            samples = np.load(join(instance_dir, f"{method}_sample_{instance}.npy"))
            numruns = len(samples)
            obj = []
            for k in range(numruns):
                obj.append(qp_eval(samples[k], Q, b))
            prob = np.mean(np.abs(obj - f_best) <= tol)
            success_prob[idx, instance] = prob

            runtime = np.load(join(instance_dir, f"{method}_runtime_{instance}.npy"))
            physical_runtime[idx, instance] = runtime

            if prob < 1e-3:
                tts = np.inf 
            elif prob > 1-1e-3:
                tts = runtime
            else:
                tts = runtime * np.ceil(np.log(1-.99)/np.log(1-prob))
            time_to_solution[idx, instance] = tts

        print(f"Instance {instance} finished.")


    success_prob_df = pd.DataFrame({})
    physical_runtime_df = pd.DataFrame({})
    time_to_solution_df = pd.DataFrame({})
    for idx in range(num_methods):
        method = methods_short[idx]
        success_prob_df[method] = success_prob[idx,:]
        physical_runtime_df[method] = physical_runtime[idx,:]
        time_to_solution_df[method] = time_to_solution[idx,:]

    save_excel_path = "qp_data"
    if not isdir(save_excel_path):
        os.mkdir(save_excel_path)
    excel_name = f"{benchmark_name}.xlsx"
    with pd.ExcelWriter(join(save_excel_path, excel_name)) as writer:  
        success_prob_df.to_excel(writer, sheet_name='success probability')
        physical_runtime_df.to_excel(writer, sheet_name='physical runtime')
        time_to_solution_df.to_excel(writer, sheet_name='time-to-solution (tts)')
    print("Dataframe saved.")
    return

if __name__ == "__main__":
    dimensions = [5, 50, 60, 75]
    resolution = 8
    tol = 1e-2
    for d in dimensions:
        save_data(d, resolution, tol)