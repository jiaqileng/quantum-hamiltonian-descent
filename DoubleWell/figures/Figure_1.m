%% QHD-Followup Figure 1: construction of instances

total_num_cells = 64;
L = 1.2;
dx = 2 * L / (total_num_cells + 1);
X = linspace(-L + dx, L-dx, total_num_cells);
Y = linspace(-L + dx, L-dx, total_num_cells);

% 1D potential
f = @(x) x.^4 - (x - 1/16).^2;
f2 = @(x1, x2) f(x1) + f(x2);
V = f(X);

figure;
plot(X, V, 'r-', 'LineWidth', 5)
ylim([-0.4, 0.6])
title('1D double well', 'Fontsize', 20)

%%
[XX, YY] = meshgrid(X, Y);
ZZ = f2(XX, YY);

figure;
surf(XX, YY, ZZ, 'FaceAlpha', 0, 'EdgeColor', 'red');
title('2D separable function', 'Fontsize', 20)

%%

L = 1.5;
dx = 2 * L / (total_num_cells + 1);
X = linspace(-L + dx, L-dx, total_num_cells);
Y = linspace(-L + dx, L-dx, total_num_cells);

[XX, YY] = meshgrid(X, Y);
ZZ = f2(XX, YY);

figure;
contour(XX, YY, ZZ, [-0.5, -0.4, -0.3, -0.2, 0, 0.2, 0.5], 'LineWidth', 3);
title('2D without rotation', 'Fontsize', 20)

%%
theta = 0.87;
c1 = cos(theta);
c2 = sin(theta);
f2_rotate = @(x1, x2) f2(c1 * x1 + c2 * x2, -c2 * x1 + c1 * x2);

ZZ_rotate = f2_rotate(XX, YY);

figure;
contour(XX, YY, ZZ_rotate, [-0.5, -0.4, -0.3, -0.2, 0, 0.2, 0.4], 'LineWidth', 3);
title('2D with rotation', 'Fontsize', 20)
