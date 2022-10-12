import cvxpy as cp
import numpy as np
#import boto3
#import io
import os

def solve_sdr(Q, b, eps):
    n = len(Q)
    X = cp.Variable((n+1,n+1), symmetric=True)
    
    # Set problem matrix in the obj
    C = np.zeros((n+1,n+1))
    C[0:n,0:n] = 0.5 * Q
    C[0:n,-1] = 0.5 * b
    C[-1,0:n] = 0.5 * b
    
    # Set constraints
    constraints = [X >> 0, X[-1, -1] == 1]
    constraints += [0 <= X[i,-1] for i in range(n)]
    constraints += [X[i,-1] <= 1 for i in range(n)]
    constraints += [X[i,i] - X[i,-1] <= 0 for i in range(n)]
    
    # Solve SDP
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(solver=cp.SCS)
    sol = X.value
    
    # Compute mu and Sigma
    mu = sol[:-1, -1]
    v = mu.reshape(n,1)
    Sigma = sol[:-1, :-1] - np.matmul(v, v.T) + eps * np.eye(n)
    
    return mu, Sigma
'''
def suggest_sampler(benchmark_name, instance, n_samples):
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
    mu, Sigma = solve_sdr(Q, b, 1e-4)
    
    samples = np.zeros((n_samples, n))
    for j in range(n_samples):
        x_suggest = np.random.multivariate_normal(mu, Sigma)
        samples[j,:] = x_suggest.clip(0,1)
        
    # Save samples to LocalResearchData
    path = f"/Users/lengjiaqi/LocalResearchData/qcqp_suggest/{benchmark_name}"
    if not os.path.isdir(path):
        os.mkdir(path)
        print("Directory '%s' created" %path)
    
    filename = f"sdr_suggest_{instance}.npy"
    np.save(os.path.join(path,filename), samples)
    return None
'''

def suggest_sampler_local(benchmark_name, instance, n_samples):
    
    # Load instance data from Local
    benchmark_path = os.path.join("/Users/lengjiaqi/LocalResearchData/qcqp_suggest", benchmark_name)
    filename = f"instance_{instance}.npy"
    with open(os.path.join(benchmark_path, filename), 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
        Q_c = np.load(f)
        b_c = np.load(f)
    
    n = len(Q)
    mu, Sigma = solve_sdr(Q, b, 1e-4)
    
    samples = np.zeros((n_samples, n))
    for j in range(n_samples):
        x_suggest = np.random.multivariate_normal(mu, Sigma)
        samples[j,:] = x_suggest.clip(0,1)
        
    # Save samples to LocalResearchData
    path = f"/Users/lengjiaqi/LocalResearchData/qcqp_suggest/{benchmark_name}"
    if not os.path.isdir(path):
        os.mkdir(path)
        print("Directory '%s' created" %path)
    
    filename = f"sdr_suggest_{instance}.npy"
    np.save(os.path.join(path,filename), samples)
    return None


if __name__ == "__main__":
	benchmark_name = "QP-5d"
	num_instances = 10
	n_samples = 1000

	for instance in range(num_instances):
	    suggest_sampler_local(benchmark_name, instance, n_samples)
