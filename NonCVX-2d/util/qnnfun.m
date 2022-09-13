function potential = qnnfun(X,Y)
% The loss function of a 2-d quantum neural network (QNN)
% Input: [X,Y] = meshgrid
% Output: potential = evaluation mesh on the input meshgrid

X_r = X.*cos(pi/7) + Y.*sin(pi/7);
Y_r = -sin(pi/7).* X + cos(pi/7).*Y;

qnn = @(x,y) .25.*((sin(2.*pi.*2.*x)-sin(pi/50)).^2 + .25.*(cos(2.*pi.*2.*x)-cos(pi/50)).^2 +...
                (sin(2.*pi.*2.*y)-sin(pi/50)).^2 + .25.*(cos(2.*pi.*2.*y)-cos(pi/50)).^2);

potential = qnn(X_r,Y_r);
end
