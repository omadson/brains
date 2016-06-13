classdef ESVM < classifier
    %ESVM     
    properties
    end
    
    methods
        function obj = ESVM(params)
            obj@classifier(params);
        end
        function obj = train(obj, inputTrain, outputTrain)
            
        end
        function outputhat = predict(obj,input)
            outputhat = 0;
        end
    end
    methods (Hidden = true)
        function out = activate(obj,val)
            if strcmp(obj.params.activeFunction, 'linear')
                out = val;
            elseif strcmp(obj.params.activeFunction, 'tanh')
                out = tanh(val);
            elseif strcmp(obj.params.activeFunction, 'RBF')
                 out = 1 ./ (1+ exp(-val));
            else
                out = val;
            end
        end
    end
end

