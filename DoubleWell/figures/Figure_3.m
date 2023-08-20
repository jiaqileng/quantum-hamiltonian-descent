%% QHD Figure 3: success probability

total_num_cells = 512;
L = 5;
dx = 2 * L / (total_num_cells + 1);
X = linspace(-L + dx, L-dx, total_num_cells);

% 1D Laplacian
e = ones(total_num_cells, 1);
D = spdiags([e -2*e e], -1:1, total_num_cells, total_num_cells);
D = full(D);
D(total_num_cells, 1) = 1;
D(1, total_num_cells, 1) = 1;
D = sparse(D);
H_D = - 0.5 * D / dx^2;
psi0 = ones(total_num_cells, 1) / sqrt(total_num_cells);

% 1D potential
f = @(x) x.^4 - (x - 1/32).^2;
d2f = @(x) 12 * x.^2 - 2;
V_diag = f(X);
H_V = spdiags([V_diag'], 0, total_num_cells, total_num_cells);

% indicator
xmin = -0.722;
delta = 0.5;
ind_nbhd = abs(X - xmin) < delta;

%% 
Lambda = [2, 4, 8, 16, 32, 64];
failure_prob = zeros(1, length(Lambda));
g_state_failure_prob = zeros(1, length(Lambda));

for k = 1:length(Lambda)
    lambda_f = Lambda(k);
    t_f = log(lambda_f);
    eps = 1 / (20 * t_f);
    
    lambda_t = @(s) exp(2 * s - t_f);
    tdep1 = @(t) 1 / lambda_t(eps * t);
    tdep2 = @(t) lambda_t(eps * t);
    T = t_f / eps;
    dt = 0.5 * dx^2 / lambda_f;
    cap_frame_every = floor(0.01 * T/dt);

    psi0 = ones(total_num_cells,1) / sqrt(total_num_cells);
    Re = abs(psi0);
    Im = zeros(total_num_cells, 1);

    [~, psi] = scheqleapfrog2(Re, Im, ...
                                           H_D, V_diag', ...
                                           T, dt, tdep1, tdep2, ...
                                           cap_frame_every);
    prob = abs(psi(end,:)).^2;
    failure_prob(k) = 1 - dot(prob, ind_nbhd);
    fprintf('lambda_f = %d, failure probability = %d\n', lambda_f, failure_prob(k));
    
    H_f = (1/lambda_f) * H_D + lambda_f * H_V;
    [eigvecs,~] = eigs(H_f, 10, 'SA');
    g_state = eigvecs(:,1);
    g_prob = abs(g_state).^2;
    g_state_failure_prob(k) = 1 - dot(g_prob, ind_nbhd);
    fprintf('lambda_f = %d, g_state failure probability = %d\n', lambda_f, g_state_failure_prob(k));
    
end
%% Plot

figure;
semilogy(Lambda, failure_prob, 'b-o', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'QHD at $t = \log(\lambda_f)$');
hold on
semilogy(Lambda, g_state_failure_prob, 'r-s', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName','Ground state at $\lambda = \lambda_f$');

legend('Interpreter','latex', 'FontSize', 16, 'Location', 'northeast')
xlabel({'$\lambda_f$'},'Interpreter','latex', 'FontSize', 14);
ylabel({'Failure probability'}, 'FontSize', 14);
%title('Failure probability', 'FontSize', 20, 'FontName', 'Helvetica')

f = gcf;
exportgraphics(f,'Figure_3.png','Resolution', 300)

