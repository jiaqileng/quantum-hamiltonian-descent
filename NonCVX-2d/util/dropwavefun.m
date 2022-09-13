function potential = dropwavefun(X,Y)
% The 2-d dropwave function
% Input: [X,Y] = meshgrid
% Output: potential = evaluation mesh on the input meshgrid

% dropwave = @(x,y) -(1 + cos(12.*sqrt((4.*x-2).^2+(4.*y-2).^2)))./...
%             (.5.*((4.*x-2).^2+(4.*y-2).^2) + 2);
        
dropwave = @(x1, x2) -(1 + cos(12*sqrt(x1.^2 + x2.^2))) ./ (0.5*(x1.^2 + x2.^2) + 2);


potential = dropwave(X,Y);
end

