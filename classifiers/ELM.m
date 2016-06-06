classdef ELM < classifier
    %ELM Classe para criação de Extreme Learning Machines
    
    properties
    end
    
    methods
        function obj = ELM(params)
            obj@classifier(params);
        end
        function obj = train(obj, inputTrain, outputTrain)
            obj.params.inputTrain  = inputTrain;
            obj.params.outputTrain = outputTrain;
            obj.params.hiddenNeurons = randn(obj.params.hiddenNeuronsNumber,size(inputTrain,2)+1);
            h = obj.activate([obj.params.hiddenNeurons * [inputTrain (-1)*ones(size(inputTrain,1),1)]']');
            obj.params.outputNeurons = pinv(h) * outputTrain;
        end
        function outputhat = predict(obj,input)
            outputhat = obj.activate([obj.params.hiddenNeurons *...
                [input (-1)*ones(size(input,1),1)]']')*obj.params.outputNeurons;
            [~,outputhat] = max(outputhat,[],2);
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

