classdef PolyTimeDep < TimeDependence
    %POLYTIMEDEP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        gamma
        pow
        
        title_fmt_str
        fname_fmt_str
    end
    
    methods
        function obj = PolyTimeDep(gamma,pow)
            %POLYTIMEDEP Construct an instance of a polynomial time
            %dependence.
            
            if nargin > 0
                obj.gamma = gamma;
                obj.pow = pow;
            else
                GAMMA_DEFAULT = 0.1;
                POW_DEFAULT = 0.1;
                obj.gamma = GAMMA_DEFAULT;
                obj.pow = POW_DEFAULT;
            end
            
            obj.title_fmt_str = "$\\phi(t) = (1 + %0.3f t^{%0.1f})^{-1}$";
            obj.fname_fmt_str = "poly_g_%0.3f_p_%0.2f";
        end
        
        function ret = eval_tdep(obj, t)
            ret = 1 ./ (1 + obj.gamma * t.^obj.pow);
        end
        
        function ret = gen_title(obj)
            %GEN_TITLE Fills the format string for the equation represented
            %by the time dependence.
            ret = sprintf(obj.title_fmt_str, obj.gamma, obj.pow);
        end
        
        function ret = gen_fname(obj)
            %GEN_FNAME Fills the format string to be used in a file name of
            %some data that was generated with this instance of the object.
            ret = sprintf(obj.fname_fmt_str, obj.gamma, obj.pow);
        end
    end
end

