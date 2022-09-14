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


def save_data(benchmark_name, num_instances, numruns, resolution, tol):
    print(benchmark_name)
    bucket_name = "amazon-braket-wugroup-us-east-1"
    
    best_fval_gurobi = np.zeros(num_instances)

    methods = ["ipopt", "snopt", "auglag", "matlab_sqp", f"post_advantage6_qaa_scheduleB_rez{resolution}", f"post_advantage6_qhd_rez{resolution}"]
    
    for method in methods:
        
        success_prob = np.zeros(num_instances)
        mean_obj = np.zeros(num_instances)
        abs_gap = np.zeros(num_instances)
        
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
        
            samples = get_samples(bucket_name, f"jiaqileng/qhd/{benchmark_name}/instance_{instance}/{method}_sample_{instance}.npy")
            
            obj = []
            for k in range(numruns):
                obj.append(qp_eval(samples[k], Q, b))
                
            success_prob[instance] = np.mean(np.abs(obj - f_best) <= tol)
            mean_obj[instance] = np.mean(obj)
            abs_gap[instance] = np.mean(obj - f_best)
        
        print(f"Method {method} finished.")
        benchmark_path = os.path.join("qp_data", benchmark_name)
        if not os.path.exists(benchmark_path):
            os.mkdir(benchmark_path)
            
        np.save(os.path.join(benchmark_path, f"{method}_success_prob.npy"), success_prob)
        np.save(os.path.join(benchmark_path, f"{method}_mean_obj.npy"), mean_obj)
        np.save(os.path.join(benchmark_path, f"{method}_abs_gap.npy"), abs_gap)
    return



# Generate QP plot data
data_path = "qp_data"
if not os.path.exists(data_path):
	os.mkdir(data_path)

save_data("QP-50d-5s", 50, 1000, 8, 1e-2)
save_data("QP-60d-5s-alt", 50, 1000, 8, 1e-2)
save_data("QP-75d-5s", 50, 1000, 8, 1e-2)