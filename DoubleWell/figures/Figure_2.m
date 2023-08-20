%% QHD-Followup Figure 2: spectral gap & ground states

total_num_cells = 1024;
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

% 1D potential
f = @(x) x.^4 - (x - 1/32).^2;
d2f = @(x) 12 * x.^2 - 2;
V_diag = f(X);
H_V = spdiags([V_diag'], 0, total_num_cells, total_num_cells);

% Solve 1st-order stationary point
syms x
eqn = 4 * x^3 - 2 * (x - 1/32);
solx = vpa(solve(eqn, x));
fprintf("First-order stationary points of f:\n");
disp(solx);
xmin = vpa(solx(3));
omega = sqrt(d2f(xmin));
fmin = f(xmin);

% interpolating parameter
g = 50;
Lambda = linspace(1, g, g);
gap = zeros(1,g);
for k = 1:g
    lambda = Lambda(k);
    H = (1/lambda^2) * H_D + H_V;
    [~, eigvals] = eigs(H, 20, 'SA', 'Tolerance', 1e-8);
    gap(k) = lambda * (eigvals(2,2) - eigvals(1,1));
end
%%

figure('position',[0,0,1200,400]);
subplot(121)
XX = linspace(-1, 1, total_num_cells);
plot(XX, f(XX) - fmin, 'r-', 'LineWidth', 5);
title({'$f(x) = x^4 - (x-1/32)^2 - c$'},'Interpreter','latex', 'FontSize', 25);

subplot(122)
plot(Lambda, gap, 'b-', 'LineWidth', 5);
hold on
plot(Lambda, ones(1, g) * omega, 'r--', 'LineWidth', 3);
legend({'$\Delta(\lambda)$', 'Semi-classical limit $\Delta(\infty) = 2.064$'},'Interpreter','latex', 'FontSize', 14, 'FontName', 'Helvetica')
ylim([0,3]);
xlabel({'$\lambda$'},'Interpreter','latex', 'FontSize', 16);
title('spectral gap $\Delta(\lambda)$','Interpreter','latex', 'FontSize', 25)

%f = gcf;
%exportgraphics(f,'semiclassical_gap.png','Resolution', 300)

%% Semiclassical approximation of the ground states
lambda = 1;
H = (1/lambda) * H_D + lambda * H_V;
[eigvecs, ~] = eigs(H, 20, 'SA');

u1 = sqrt(1 / dx) * eigvecs(:,1);
v1 = (lambda * omega / pi)^(1/4) * exp(- omega * lambda * (X - xmin).^2 ./ 2);

lambda = 40;
H = (1/lambda) * H_D + lambda * H_V;
[eigvecs, ~] = eigs(H, 20, 'SA');

u40 = sqrt(1 / dx) * eigvecs(:,1);
v40 = (lambda * omega / pi)^(1/4) * exp(- omega * lambda * (X - xmin).^2 ./ 2);

figure('position',[0,0,1200,400]);
subplot(121)
plot(X, abs(u1), 'b-', 'LineWidth', 2);
hold on
plot(X, abs(v1), 'r--', 'LineWidth', 2)
plot(xmin * ones(2), [0, 1], 'k-', 'LineWidth', 2)
text(-3, 0.95, '$x_{\min} = -0.72$', 'Interpreter','latex', 'FontSize', 14)
title({'$\lambda = 1$'},'Interpreter','latex', 'FontSize', 25)
legend({'Ground state', 'Semi-classical approx.'}, 'FontSize', 14, 'FontName', 'Helvetica');

subplot(122)
plot(X, abs(u40), 'b-', 'LineWidth', 2);
hold on
plot(X, abs(v40), 'r--', 'LineWidth', 2)
plot(xmin * ones(2), [0, 2.5], 'k-', 'LineWidth', 2)
text(-3, 2.4, '$x_{\min} = -0.72$', 'Interpreter','latex', 'FontSize', 14)
title({'$\lambda = 40$'},'Interpreter','latex', 'FontSize', 25)
legend({'Ground state', 'Semi-classical approx.'}, 'FontSize', 14, 'FontName', 'Helvetica');

%f = gcf;
%exportgraphics(f,'semiclassical_state.png','Resolution',300)