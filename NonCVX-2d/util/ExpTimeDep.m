classdef ExpTimeDep < TimeDependence
    %EXPTIMEDEP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        gamma
        
        title_fmt_str
        fname_fmt_str
    end
    
    methods
        function obj = ExpTimeDep(gamma)
            %EXPTIMEDEP Construct an instance of a exponential time
            %dependence.
            
            if nargin > 0
                obj.gamma = gamma;
            else
                GAMMA_DEFAULT = 0.1;
                obj.gamma = GAMMA_DEFAULT;
            end
            
            obj.title_fmt_str = "$\\phi(t)=\\exp(-%0.3f t)$";
            obj.fname_fmt_str = "exp_g_%0.3f";
        end
        
        function ret = eval_tdep(obj, t)
            ret = exp(-obj.gamma * t);
        end
        
        function ret = gen_title(obj)
            %GEN_TITLE Fills the format string for the equation represented
            %by the time dependence.
            ret = sprintf(obj.title_fmt_str, obj.gamma);
        end
        
        function ret = gen_fname(obj)
            %GEN_FNAME Fills the format string to be used in a file name of
            %some data that was generated with this instance of the object.
            ret = sprintf(obj.fname_fmt_str, obj.gamma);
        end
    end
end

