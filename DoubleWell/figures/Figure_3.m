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

%%
lambda_f = 40;
H_f = (1/lambda_f) * H_D + lambda_f * H_V;
EPS = [5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1, 5e-1];
ERR = zeros(1, length(EPS));

[eigvecs,~] = eigs(H_f, 10, 'SA');
g_state = eigvecs(:,1);

%%
for k = 1:length(EPS)
    eps = EPS(k);
    err = ground_state_error(eps, lambda_f, H_D, V_diag, dx, total_num_cells, g_state);
    fprintf('eps = %d, infidelity = %d\n', eps, err);
    ERR(k) = err;
    endle

%% Polyfit & plot

figure;
loglog(EPS, ERR, 'b-o', 'LineWidth', 2, 'DisplayName', 'Numerical results');
hold on
loglog(EPS, EPS, 'r--', 'LineWidth', 2, 'DisplayName','Theoretical ref: $y = \epsilon$');

legend('Interpreter','latex', 'FontSize', 14, 'Location', 'northwest')
xlabel({'$\epsilon$'},'Interpreter','latex', 'FontSize', 14)
title({'Adiabatic error v.s. $\epsilon$'},'Interpreter','latex', 'FontSize', 16)

%f = gcf;
%exportgraphics(f,'semiclassical_gap.png','Resolution', 300)

%%
function err = ground_state_error(eps, lambda_f, H_D, V_diag, dx, total_num_cells, g_state)
%%
%eps = 5e-2;
%lambda_f = 40;

t_f = log(lambda_f);
lambda_t = @(s) exp(2 * s - t_f);

tdep1 = @(t) 1 / lambda_t(eps * t);
tdep2 = @(t) lambda_t(eps * t);
T = t_f / eps;
dt = 0.5 * dx^2 / lambda_f;
cap_frame_every = floor(0.01 * T/dt);

[psi0, ~] = eigs(H_D, 1, 'SA');
Re = abs(psi0);
Im = zeros(total_num_cells, 1);

[~, psi] = scheqleapfrog2(Re, Im, ...
                                       H_D, V_diag', ...
                                       T, dt, tdep1, tdep2, ...
                                       cap_frame_every);

err = 1 - abs(dot(psi(end,:)', g_state)).^2;
end