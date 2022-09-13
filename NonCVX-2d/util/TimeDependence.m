classdef (Abstract) TimeDependence
    %TIMEDEPENDENCE Abstract base class for time dependence classes.
    %   Since time dependences may have an unknown number of parameters, it
    %   is easier to subclass them individually.
    
    properties (Abstract)
        title_fmt_str
        fname_fmt_str
    end
    
    methods (Abstract)
        eval_tdep(obj, t);
        gen_title(obj);
        gen_fname(obj);
    end
end

