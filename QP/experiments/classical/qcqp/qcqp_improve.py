import numpy as np
from numpy.random import randn
import cvxpy as cvx
from qcqp import *
import boto3
import io
import os
import multiprocessing
from joblib import Parallel, delayed
from itertools import product

def assign_vars(xs, vals):
    if vals is None:
        for x in xs:
            size = x.size[0]*x.size[1]
            x.value = np.full(x.size, np.nan)
    else:
        ind = 0
        for x in xs:
            size = x.size[0]*x.size[1]
            x.value = np.reshape(vals[ind:ind+size], x.size, order='F')
            ind += size


def improve_by_coord_descent(benchmark_name, instance):
    '''
    s3 = boto3.resource('s3')
    bucket_name = "amazon-braket-wugroup-us-east-1"
    bucket = s3.Bucket(bucket_name)


    # Load instance data from S3
    instance_filename = f"jiaqileng/qhd/{benchmark_name}/instance_{instance}/instance_{instance}.npy"
    res = boto3.client('s3').get_object(Bucket=bucket_name, Key=instance_filename)
    bytes_ = io.BytesIO(res['Body'].read())
    Q = np.load(bytes_)
    b = np.load(bytes_)
    Q_c = np.load(bytes_)
    b_c = np.load(bytes_)
    n = len(Q)
    '''
    # Local instance data from Local
    benchmark_path = os.path.join("/Users/lengjiaqi/LocalResearchData/qcqp_suggest", benchmark_name)
    filename = f"instance_{instance}.npy"
    with open(os.path.join(benchmark_path, filename), 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
        Q_c = np.load(f)
        b_c = np.load(f)
    n = len(Q)

    # Load suggest samples from Local
    path = f"/Users/lengjiaqi/LocalResearchData/qcqp_suggest/{benchmark_name}"
    suggest_filename = f"sdr_suggest_{instance}.npy"
    suggest_samples = np.load(os.path.join(path, suggest_filename))
    num_samples = len(suggest_samples)

    # Form a nonconvex problem.
    x = cvx.Variable(n)
    obj = 0.5 * cvx.quad_form(x, Q) + b @ x
    cons = [0 <= x, x <= 1]
    prob = cvx.Problem(cvx.Minimize(obj), cons)

    # Create a QCQP handler.
    qcqp = QCQP(prob)

    # Improve the suggest using qcqp
    improve_samples = np.zeros((num_samples, n))
    for idx in range(num_samples):
        x_suggest = suggest_samples[idx]
        assign_vars(qcqp.prob.variables(), x_suggest)
        f_cd, v_cd = qcqp.improve(COORD_DESCENT)
        #print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
        x_improve = x.value.reshape(n)
        improve_samples[idx] = x_improve
        if idx % 100 == 0:
            print(f"Benchmark: {benchmark_name}, instance {instance}: {idx}-th improvment has finished.")

    # Save improve samples to Local
    improve_filename = f"qcqp_{instance}.npy"
    np.save(os.path.join(path,improve_filename), improve_samples)
    
    return None


if __name__ == "__main__":
    #benchmark = ["QP-50d-5s", "QP-60d-5s-alt", "QP-75d-5s"]
    benchmark = ["QP-5d"]
    num_instances = 10
    num_cores = multiprocessing.cpu_count()
    print(f'Num. of cores: {num_cores}.')

    par_list = Parallel(n_jobs=num_cores)(delayed(improve_by_coord_descent)(name, tid) for name, tid in product(benchmark, range(num_instances)))

