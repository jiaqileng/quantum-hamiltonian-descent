import numpy as np
import os, io
from io import BytesIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import boto3
import pandas as pd
import json

def qp_eval(x, Q, b):
    return 0.5 * x @ Q @ x + b @ x

def get_samples(bucket_name, filename):
    res = boto3.client('s3').get_object(Bucket=bucket_name, Key=filename)
    bytes_ = io.BytesIO(res['Body'].read())
    return np.load(bytes_)


def save_data(benchmark_name, num_instances, resolution, tol):
    print(benchmark_name)
    bucket_name = "amazon-braket-wugroup-us-east-1"

    methods = [f"tnc_post_advantage6_qhd_rez{resolution}", "snopt",
               f"tnc_post_advantage6_qaa_scheduleB_rez{resolution}", "ipopt", "tnc", "matlab_sqp", "qcqp",]
    methods_short = ["DW-QHD","SNOPT","DW-QAA","IPOPT","TNC","MATLAB","QCQP"]
    num_methods = len(methods)
    success_prob = np.zeros((num_methods, num_instances))
    physical_runtime = np.zeros((num_methods, num_instances))
    time_to_solution = np.zeros((num_methods, num_instances))

    for instance in range(num_instances):

        # Load instance data from S3
        instance_filename = f"jiaqileng/qhd/{benchmark_name}/instance_{instance}/instance_{instance}.npy"
        res = boto3.client('s3').get_object(Bucket=bucket_name, Key=instance_filename)
        bytes_ = io.BytesIO(res['Body'].read())
        Q = np.load(bytes_)
        b = np.load(bytes_)
        Q_c = np.load(bytes_)
        b_c = np.load(bytes_)

        # Load Gurobi data (ground truth)
        gurobi_filename = f"jiaqileng/qhd/{benchmark_name}/instance_{instance}/gurobi_solution_{instance}.npy"
        res = boto3.client('s3').get_object(Bucket=bucket_name, Key=gurobi_filename)
        bytes_ = io.BytesIO(res['Body'].read())
        x_best = np.load(bytes_)
        f_best = qp_eval(x_best, Q, b)

        for idx in range(num_methods):
            method = methods[idx]
 
            samples = get_samples(bucket_name, f"jiaqileng/qhd/{benchmark_name}/instance_{instance}/{method}_sample_{instance}.npy")
            numruns = len(samples)
            obj = []
            for k in range(numruns):
                obj.append(qp_eval(samples[k], Q, b))
            prob = np.mean(np.abs(obj - f_best) <= tol)
            success_prob[idx, instance] = prob

            runtime = get_samples(bucket_name, f"jiaqileng/qhd/{benchmark_name}/instance_{instance}/{method}_runtime_{instance}.npy")
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
    excel_name = f"{benchmark_name}.xlsx"
    with pd.ExcelWriter(os.path.join(save_excel_path, excel_name)) as writer:  
        success_prob_df.to_excel(writer, sheet_name='success probability')
        physical_runtime_df.to_excel(writer, sheet_name='physical runtime')
        time_to_solution_df.to_excel(writer, sheet_name='time-to-solution (tts)')
    print("Dataframe saved.")
    return

if __name__ == "__main__":
    dimensions = [50, 60, 75]
    num_instances = 50
    resolution = 8
    tol = 1e-2
    for d in dimensions:
        benchmark_name = f"QP-{d}d-5s"
        save_data(benchmark_name, num_instances, resolution, tol)