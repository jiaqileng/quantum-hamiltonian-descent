format compact;
% Set path (from snopt files)
%setpath;

% Problem specifications to modify
numruns = 1000;
NUM_INSTANCES = 50;

% Please change the data directory and benchmark name
DATA_DIR = "/Users/lengjiaqi/QHD_DATA/NonCVX-2d"; 
benchmark_name = "QP-75d-5s";

% Input directory of mat files
directory_name = fullfile(DATA_DIR, benchmark_name);

%strcat("matlab-", instances_name);
if ~exist(directory_name, 'dir')
    mkdir(directory_name)
end

%%
for instance=0:NUM_INSTANCES-1
    instance_dir = fullfile(directory_name, strcat("instance_", int2str(instance)));
    % Load problem instance
    load(fullfile(instance_dir, strcat("instance_", int2str(instance), ".mat")))
    % Load random initialization
    load(fullfile(instance_dir, strcat("rand_init_", int2str(instance), ".mat")))
    dimension = size(Q,1);
    matlab_sample = zeros(numruns, dimension);
    fun = @(x) (transpose(x) * Q * x) / 2 + transpose(b) * x;
    options = optimoptions("fmincon", "Algorithm", "sqp");
    for i=1:numruns
        x0 = transpose(rand_init(i,:));
        matlab_sample(i,:) = fmincon(fun, x0, [], [], Q_c, b_c, zeros(dimension,1), ones(dimension,1), [], options);
    end
        
    filename = fullfile(instance_dir, strcat("matlab_sqp_sample_", int2str(instance), ".mat"));
    save(filename, "matlab_sample")
end
