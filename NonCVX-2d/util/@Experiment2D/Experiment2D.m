classdef Experiment2D
    %EXPERIMENT2D Specification for a 2-dimensional experiment
    %   Contains the potential function, domain information
    
    properties
        % String: Directory name for the function
        experiment_dir
        
        % String: Proper name of the function, used in plot titles or
        % debugging or error handling statements
        experiment_V_str
        
        % The original domain is (x1, x2) = bounds x bounds
        bounds
        % The new domain that we've normalized to
        normalized_bounds
        
        % Edge length of the original square domain
        L
        
        % Number of cells in the discretization of each dimension
        cells_along_dim
        
        % Size of each grid cell
        step_size
        
        % Meshgrid over dimensions, will be normalized
        X1
        X2
        
        % Function handle to the original function
        potential_fn
        
        % Location of the global minimum
        global_min
        
        % Value of the exact global minimum, not the one tied to the grid.
        % Used to calculate the function value after normalization to shift
        % the global min to 0.
        global_min_val
        
        % Positive scalar: defines a circular neighborhood around the global minimizer
        nbhd_radius
       
        % (Approximated) Lipschitz constant of the function
        lipschitz
        
        % Objective function evaluated over the meshgrid
        V 
        
        % Kinetic Hamiltonian
        H_T
        
        % The main diagonal of the potential Hamiltonian
        H_U
        
        % Indices on the mesh within the neighborhood of the global minimizer
        mesh_ind
        
        % Main diagonal of the index matrix of the "neighborhood" of the global minimizer
        H_ind 
        
        sym_gradient
        symbols_in_grad
        gradient
        
        sym_hessian
        symbols_in_hess
        hessian
    end
    
    methods
        function obj = Experiment2D(experiment_dir, experiment_V_str, bounds, global_min,... 
                                        cells_along_dim, potential_fn, nbhd_radius)
            %Experiment2D Constructor
            %   Populates the instance variables of an experiment, allowing
            %   the experiment object to be either accessed in another script or
            %   used directly by instance methods. We define instance
            %   methods to run numerical integration forwards in time,
            %   reducing the amount of boilerplate code required to define
            %   new experiment scripts.
            
            % Strings for filesystem and plot management
            obj.experiment_dir = experiment_dir;
            obj.experiment_V_str = experiment_V_str;

            obj.bounds = bounds;

            % Length along each edge (the domain is always square in our experiments)
            obj.L = bounds(2) - bounds(1);

            obj.cells_along_dim = cells_along_dim;

            % Function handle to the (possibly) unnormalized function.
            % If an objective function is already defined on (x1, x2)=[0, 1]x[0, 1]
            % then the normalization steps will not scale the function.
            % If the global minimum (minima) is (are) not zero, the function 
            % will be shifted so that it becomes zero.
            obj.potential_fn = potential_fn;

            % Original function specification
            original_step_size = obj.L / (obj.cells_along_dim+1);
            original_X1 = bounds(1) + original_step_size : original_step_size : bounds(2) - original_step_size;
            original_X2 = original_X1;
            [original_X1, original_X2] = meshgrid(original_X1, original_X2);
            original_V = obj.potential_fn(original_X1, original_X2);

            % Normalizing the function to the new domain: (x1, x2)=[0, 1]x[0, 1]
            obj.normalized_bounds = [0, 1];
            obj.step_size = 1 / (obj.cells_along_dim + 1);
            X1 = obj.step_size : obj.step_size : 1 - obj.step_size;
            X2 = obj.step_size : obj.step_size : 1 - obj.step_size;
            [obj.X1, obj.X2] = meshgrid(X1, X2);

            % Normalize the location of the global minimum.
            obj.global_min = [(global_min(1) - bounds(1))/obj.L, (global_min(2) - bounds(1))/obj.L];
            
            % Calculate the value of the global minimum prior to any shift.
            % Note global_min is the input parameters, obj.global_min is
            % the normalized location of the global min.
            obj.global_min_val = obj.potential_fn(global_min(1), global_min(2));

            % Scale the neighborhood radius along with the domain.
            obj.nbhd_radius = nbhd_radius/obj.L;

            % Scale the function value along with the scaled domain to
            % preserve curvature. Also shift the function so its global
            % minimum is equal to zero. The grid points
            % remain sampled in the same locations relative to the bounds
            % of the original domain.
            obj.V = (original_V - obj.global_min_val) / obj.L;

            % Approximate the Lipschitz constant, which is invariant 
            % under normalization here.
            [FX, FY] = gradient(obj.V, obj.step_size);
            grad_norm = sqrt(FX.^2 + FY.^2);
            obj.lipschitz = max(grad_norm, [], 'all');

            % H_T and H_U are normalized here
            obj.H_T = (-0.5 / (obj.step_size)^2) .* laplacian2d(cells_along_dim);
            obj.H_U = obj.V(:);

            % Obtains indices within the neighborhood radius
            % (nbhd_rad) of the global minimum.
            obj.mesh_ind = (obj.X1 - obj.global_min(1)).^2 + (obj.X2 - obj.global_min(2)).^2 <= obj.nbhd_radius^2;
            obj.H_ind = double(obj.mesh_ind(:));
            
            % Precompute symbolic gradient functions along both dimensions.
            syms x1 x2

            % The symbolic gradient may depend on zero, one or both 
            % variables. The gradient function must be wrapped to accept 
            % two variables either way, as there's no easy way to know 
            % beforehand.
            obj.sym_gradient = gradient(obj.potential_fn(x1, x2), [x1, x2]);
            core_gradient = matlabFunction(obj.sym_gradient);
            obj.gradient = matlabFunction(obj.sym_gradient);
            obj.symbols_in_grad = symvar(obj.sym_gradient);
            
            if numel(obj.symbols_in_grad) == 2
                obj.gradient = @(x1v, x2v) core_gradient(x1v, x2v);
            elseif numel(obj.symbols_in_grad) == 1
                if isequal(obj.symbols_in_grad(1), x1)
                    obj.gradient = @(x1v, x2v) core_gradient(x1v);
                elseif isequal(obj.symbols_in_grad(1), x2)
                    obj.gradient = @(x1v, x2v) core_gradient(x2v);
                else
                    error("Failure in symbolic manipulation of gradient dependent on one variable");
                end
            elseif numel(obj.symbols_in_grad) == 0
                % If the gradient is constant
                obj.gradient = @(x1v, x2v) core_gradient();
            else
                error("Failure in gradient symbolic calculation");
            end
            
            obj.sym_hessian = hessian(obj.potential_fn(x1, x2), [x1, x2]);
            core_hessian = matlabFunction(obj.sym_hessian);
            obj.symbols_in_hess = symvar(obj.sym_hessian);
            
            if numel(obj.symbols_in_hess) == 2
                obj.hessian = @(x1v, x2v) core_hessian(x1v, x2v);
            elseif numel(obj.symbols_in_hess) == 1
                if isequal(obj.symbols_in_hess(1), x1)
                    obj.hessian = @(x1v, x2v) core_hessian(x1v);
                elseif isequal(obj.symbols_in_hess(1), x2)
                    obj.hessian = @(x1v, x2v) core_hessian(x2v);
                else
                    error("Failure in symbolic manipulation of Hessian dependent on one variable");
                end
            elseif numel(obj.symbols_in_hess) == 0
                % If the Hessian is constant
                obj.hessian = @(x1v, x2v) core_hessian();
            else
                error("Failure in Hessian symbolic calculation");
            end
        end
        
        
        % Instance Methods
        
        function val = eval_fn(obj, x1, x2)
            % eval_fn evaluates the normalized function value at (x1,x2).

            x1 = obj.bounds(1) + x1 * obj.L;
            x2 = obj.bounds(1) + x2 * obj.L;
            
            % Shift the function by the global min.
            val = (obj.potential_fn(x1, x2) - obj.global_min_val) / obj.L;
        end
        
        
        function g = eval_grad(obj, x1v, x2v)
            % eval_grad evaluates the gradient at a given point.            
            x1v = obj.bounds(1) + x1v * obj.L;
            x2v = obj.bounds(1) + x2v * obj.L;
            
            g = obj.gradient(x1v, x2v);
        end
        
        
        function H = eval_hessian(obj, x1v, x2v)
            % eval_hessian evaluates the Hessian at a given point.
            syms x1 x2;
            
            x1v = obj.bounds(1) + x1v * obj.L;
            x2v = obj.bounds(1) + x2v * obj.L;

            H = obj.hessian(x1v, x2v);
        end
        
        
        function plot_fn(obj)
            % plot_fn creates a surface plot of the objective function and a
            % red circle indicating the neighborhood of the global minimizer.

            figure;
            surfc(obj.X1, obj.X2, obj.V);
            shading interp
            hold on
            contour(obj.X1, obj.X2, double(obj.mesh_ind),'r-','LineWidth', 3);
            title(obj.experiment_V_str);
        end
        
        function b = is_pt_in_neighborhood(obj, point)                        
            dist_from_global_min = norm(point - obj.global_min);
            if dist_from_global_min < obj.nbhd_radius
                b = true;
            else
                b = false;
            end
        end
        
        % Instance methods defined in separate files in the class folder.        
        [snapshot_times, psi] = fdmleapfrog(obj, init_type, T, tdep, tdep2, cap_frame_every)
 
        [snapshot_times, psi] = pseudospec(obj, init_type, T, tdep1, tdep2, cap_frame_every)
        
        [snapshot_times, pos, vel, obs] = classicalLeapfrog(obj, x_0, v_0, v_cutoff, tdep, cap_frame_every, max_frames)
    end  % Methods
end  % Experiment2D
