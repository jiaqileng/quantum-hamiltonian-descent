% Optimization Test Functions

% Uncomment num_cells to allow this script to be run on its own.
% num_cells = 256;


% Ackley No. 2 Function
ackley2_fn = @(x1, x2) -200 * exp(-0.2*sqrt(x1.^2 + x2.^2));
ackley2_bounds = [-32, 32];
ackley2_global_min = [0,0];
ackley2_radius = 2;
ackley2_experiment = Experiment2D("ackley2", "Ackley No. 2 Function",...
    ackley2_bounds, ackley2_global_min, num_cells, ackley2_fn, ackley2_radius);

% Ackley Function
% https://www.sfu.ca/~ssurjano/ackley.html
ackley_bounds = [-32.768, 32.768];
ackley_global_min = [0, 0];
a = 20;
b = 0.2;
c = 2*pi;
ackley_fn = @(x1, x2) -a * exp(-b * sqrt(0.5 * (x1.^2 + x2.^2))) - exp(0.5 * (cos(c * x1) + cos(c * x2))) + a + exp(1);
ackley_radius = 2;
ackley_experiment = Experiment2D("ackley", "Ackley function",...
    ackley_bounds, ackley_global_min, num_cells, ackley_fn, ackley_radius);

% Alpine No. 1 Function
alpine1_fn = @(x1, x2) abs(x1 .* sin(x1) + 0.1*x1) + abs(x2 .* sin(x2) + 0.1*x2);
alpine1_bounds = [-10, 10];
alpine1_global_min = [0, 0];
alpine1_radius = 2;
alpine1_experiment = Experiment2D("alpine1", "Alpine No. 1 function",....
    alpine1_bounds, alpine1_global_min, num_cells, alpine1_fn, alpine1_radius);

% Alpine No. 2 Function
% We flip the function for minimization.
alpine2_fn = @(x1, x2) -sqrt(x1) .* sin(x1) .* sqrt(x2) .* sin(x2);
alpine2_bounds = [0, 10];
alpine2_global_min = [7.9171, 7.9171];
alpine2_radius = 1.5;
alpine2_experiment = Experiment2D("alpine2", "Alpine 2 function",...
    alpine2_bounds, alpine2_global_min, num_cells, alpine2_fn, alpine2_radius);

% Bohachevsky No. 2 Function
bohachevsky2_fn = @(x1, x2) x1.^2 + 2*x2.^2 - (0.3 * cos(3*pi * x1) .* cos(4*pi * x2)) + 0.3;
bohachevsky2_bounds = [-5, 5];
bohachevsky2_global_min = [0, 0];
bohachevsky2_radius = 1;
bohachevsky2_experiment = Experiment2D("bohachevsky2", "Bohachevsky No. 2 function",...
    bohachevsky2_bounds, bohachevsky2_global_min, num_cells, bohachevsky2_fn, bohachevsky2_radius);

% Camel Function - 3 hump
camel3_fn = @(x1, x2) 2*x1.^2 - 1.05*x1.^4 + (x1.^6)/6 + x1.*x2 + x2.^2;
camel3_bounds = [-2, 2];
camel3_global_min = [0, 0];
camel3_radius = 0.5;
camel3_experiment = Experiment2D("camel3", "Camel 3 function",...
    camel3_bounds, camel3_global_min, num_cells, camel3_fn, camel3_radius);

% Csendes Function
csendes_fn = @(x1, x2) x1.^6 .* (2 + sin(1./x1)) + x2.^6 .* (2 + sin(1./x2));
csendes_bounds = [-1, 1];
csendes_global_min = [1e-6, 1e-6];
csendes_radius = 0.25;
csendes_experiment = Experiment2D("csendes", "Csendes function",...
    csendes_bounds, csendes_global_min, num_cells, csendes_fn, csendes_radius);

% Deflected Corrugated Spring Function
a = 5;
K = 5;
defl_corr_spring_fn = @(x1, x2) 0.1*((x1 - a).^2 + (x2 - a).^2) - cos(K * sqrt((x1 - a).^2 + (x2 - a).^2));
defl_corr_spring_bounds = [0, 10];
defl_corr_spring_global_min = [5, 5];
defl_corr_spring_radius = 1;
defl_corr_spring_experiment = Experiment2D("defl_corr_spring", "Deflected Corrugated Spring function",...
    defl_corr_spring_bounds, defl_corr_spring_global_min, num_cells, defl_corr_spring_fn, defl_corr_spring_radius);

% Drop Wave
% https://www.sfu.ca/~ssurjano/grlee12.html
dropwave_fn = @(x1, x2) -(1 + cos(12*sqrt(x1.^2 + x2.^2))) ./ (0.5*(x1.^2 + x2.^2) + 2);
dropwave_bounds = [-5.12, 5.12];
dropwave_global_min = [0, 0];
dropwave_radius = 2;
dropwave_experiment = Experiment2D("dropwave", "Drop-Wave function",...
    dropwave_bounds, dropwave_global_min, num_cells, dropwave_fn, dropwave_radius);

% Easom Function
% https://www.sfu.ca/~ssurjano/easom.html
easom_fn = @(x1, x2) -cos(x1) .* cos(x2) .* exp(-x1.^2 - x2.^2);
easom_bounds = [-10, 10];
easom_global_min = [0, 0];
easom_radius = 1;
easom_experiment = Experiment2D("easom", "Easom function",...
    easom_bounds, easom_global_min, num_cells, easom_fn, easom_radius);

% Griewank Function
% https://www.sfu.ca/~ssurjano/griewank.html
griewank_fn = @(x1, x2) (x1.^2 + x2.^2)./4000 - cos(x1).*cos(x2/sqrt(2)) + 1;
griewank_bounds = [-10, 10];
griewank_global_min = [0, 0];
griewank_radius = 2;
griewank_experiment = Experiment2D("griewank", "Griewank function",...
    griewank_bounds, griewank_global_min, num_cells, griewank_fn, griewank_radius);

% Holder Table Function
% https://www.sfu.ca/~ssurjano/holder.html
holder_fn = @(x1, x2) - abs(sin(x1) .* cos(x2) .* exp(abs(1 - sqrt(x1.^2 + x2.^2)/pi)));
holder_bounds = [0, 10];
holder_global_min = [8.05502, 9.66459];
holder_radius = 1.5;
holder_experiment = Experiment2D("holder", "Holder Table function",...
    holder_bounds, holder_global_min, num_cells, holder_fn, holder_radius);

% Hosaki Function
hosaki_fn = @(x1, x2) (1 - 8*x1 + 7*x1.^2 - (7/3)*x1.^3 + (1/4)*x1.^4) .* x2.^2 .* exp(-x2);
hosaki_bounds = [0, 5];
hosaki_global_min = [4, 2];
hosaki_radius = 0.5;
hosaki_experiment = Experiment2D("hosaki", "Hosaki function",...
    hosaki_bounds, hosaki_global_min, num_cells, hosaki_fn, hosaki_radius);

% Levy Function
% https://www.sfu.ca/~ssurjano/levy.html
w = @(v) 1 + (1/4)*(v - 1);
levy_fn = @(x1, x2) ...
    sin(pi * w(x1)).^2 ...
    + (w(x1) - 1).^2 .* (1 + 10*sin(pi*w(x1) + 1).^2) ...
    + (w(x2) - 1).^2 .* (1 + sin(2*pi*w(x2)).^2);
levy_bounds = [-10, 10];
levy_global_min = [1, 1];
levy_radius = 2;
levy_experiment = Experiment2D("levy", "Levy Function",...
    levy_bounds, levy_global_min, num_cells, levy_fn, levy_radius);

% Levy Function N. 13
% https://www.sfu.ca/~ssurjano/levy13.html
% Scaled by 0.05
levy13_fn = @(x1, x2) ...
    0.05 * ( ...
        sin(3*pi * x1).^2 ...
        + ((x1 - 1).^2 .* (1 + sin(3*pi * x2).^2)) ...
        + ((x2 - 1).^2 .* (1 + sin(2*pi * x2).^2)) ...
    );
levy13_bounds = [-10, 10];
levy13_global_min = [1, 1];
levy13_radius = 2;
levy13_experiment = Experiment2D("levy13", "Levy function No. 13",...
    levy13_bounds, levy13_global_min, num_cells, levy13_fn, levy13_radius);

% Michalewicz Function
% https://www.sfu.ca/~ssurjano/michal.html
m = 10;
param_michalewicz_fn = @(x1, x2, m) -( (sin(x1) .* sin(1 * x1.^2 / pi).^(2*m)) + (sin(x2) .* sin(2 * x2.^2 / pi).^(2*m)) );
michalewicz_fn = @(x1, x2) param_michalewicz_fn(x1, x2, m);
michalewicz_bounds = [0, pi];
michalewicz_global_min = [2.2, 1.57];
michalewicz_radius = 0.3;
michalewicz_experiment = Experiment2D("michalewicz", "Michalewicz function",...
    michalewicz_bounds, michalewicz_global_min, num_cells, michalewicz_fn, michalewicz_radius);

% Rastrigin Function
% https://www.sfu.ca/~ssurjano/rastr.html
rastrigin_fn = @(x1, x2) 20 + (x1.^2 - 10*cos(2*pi * x1)) + (x2.^2 - 10*cos(2*pi * x2));
rastrigin_bounds = [-5.12, 5.12];
rastrigin_global_min = [0, 0];
rastrigin_radius = 1.5;
rastrigin_experiment = Experiment2D("rastrigin", "Rastrigin function",...
    rastrigin_bounds, rastrigin_global_min, num_cells, rastrigin_fn, rastrigin_radius);

% Rosenbrock Function
% https://www.sfu.ca/~ssurjano/rosen.html
% https://en.wikipedia.org/wiki/Rosenbrock_function
% Scaled by 1/100
rosenbrock_bounds = [-1.5, 1.5];
rosenbrock_global_min = [1, 1];
rosenbrock_fn = @(x1, x2) (x2 - x1.^2).^2 + (1/100)*(1 - x1).^2;
rosenbrock_radius = 0.75;
rosenbrock_experiment = Experiment2D("rosenbrock", "Rosenbrock function",...
    rosenbrock_bounds, rosenbrock_global_min, num_cells, rosenbrock_fn, rosenbrock_radius);

% Shubert function
% https://www.sfu.ca/~ssurjano/shubert.html
% shubert_fn = @(x1, x2) ...
%     (cos(2*x1 + 1) + 2*cos(3*x1 + 2) + 3*cos(4*x1 + 3) + 4*cos(5*x1 + 4) + 5*cos(6*x1 + 5)) ...
%     .* (cos(2*x2 + 1) + 2*cos(3*x2 + 2) + 3*cos(4*x2 + 3) + 4*cos(5*x2 + 4) + 5*cos(6*x2 + 5));

% Use a lightly modified Shubert function to restrict the domain to include
% only one global minimum.
shubert_fn = @(x1, x2) ...
    (cos(2*x1 + 1) + 2*cos(3*x1 + 2) + 3*cos(4*x1 + 3)) ...
    .* (cos(2*x2 + 1) + cos(x2 + 2));

shubert_bounds = [-2, 2];
shubert_global_min = [-0.7146, 1.0850];
shubert_radius = 0.5;
shubert_experiment = Experiment2D("shubert", "Shubert function",...
    shubert_bounds, shubert_global_min, num_cells, shubert_fn, shubert_radius);

% Styblinski-Tang
% https://www.sfu.ca/~ssurjano/stybtang.html
% Scaled by 1/78
styblinski_tang_fn = @(x1, x2) 1/78 * 0.5 * (x1.^4 - 16*x1.^2 + 5*x1 + x2.^4 - 16*x2.^2 + 5*x2);
styblinski_tang_bounds = [-5, 5];
styblinski_tang_global_min = [-2.9035, -2.9035];
styblinski_tang_radius = 1;
styblinski_tang_experiment = Experiment2D("styblinski_tang", "Styblinski-Tang function",...
    styblinski_tang_bounds, styblinski_tang_global_min, num_cells, styblinski_tang_fn, styblinski_tang_radius);

% Sum of Squares Function
sumofsquares_fn = @(x1, x2) x1.^2 + 2*x2.^2;
sumofsquares_bounds = [-1, 1];
sumofsquares_global_min = [0, 0];
sumofsquares_radius = 0.2;
sumofsquares_experiment = Experiment2D("sumofsquares", "Sum of Squares function", ...
    sumofsquares_bounds, sumofsquares_global_min, num_cells, sumofsquares_fn, sumofsquares_radius);

% Sum of Squares Function
B = 15;
m = 3;
xinsheyang3_fn = @(x1, x2) ...
    exp(-(x1 ./ B).^(2*m) - (x2 ./ B).^(2*m)) - 2*exp(-x1.^2 - x2.^2) .* cos(x1).^2 .* cos(x2).^2;
xinsheyang3_bounds = [-20, 20];
xinsheyang3_global_min = [0, 0];
xinsheyang3_radius = 1;
xinsheyang3_experiment = Experiment2D("xinsheyang3", "Xin-She Yang 3 function", ...
    xinsheyang3_bounds, xinsheyang3_global_min, num_cells, xinsheyang3_fn, xinsheyang3_radius);


experiments = [...
    ackley_experiment, ...
    ackley2_experiment, ...
    alpine1_experiment, ...
    alpine2_experiment, ...
    bohachevsky2_experiment, ...
    camel3_experiment, ...
    csendes_experiment, ...
    defl_corr_spring_experiment, ...
    dropwave_experiment, ...
    easom_experiment, ...
    griewank_experiment, ...
    holder_experiment, ...
    hosaki_experiment, ...
    levy_experiment, ...
    levy13_experiment, ...
    michalewicz_experiment, ...
    rastrigin_experiment, ...
    rosenbrock_experiment, ...
    shubert_experiment, ...
    styblinski_tang_experiment, ...
    sumofsquares_experiment, ...
    xinsheyang3_experiment, ...
];