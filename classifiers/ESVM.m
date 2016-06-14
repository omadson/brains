classdef ESVM < classifier
    %ESVM     
    properties
    end
    
    methods
        function obj = ESVM(params)
            obj@classifier(params);
        end
        function obj = train(obj, inputTrain, outputTrain)
            obj.params.inputTrain = inputTrain;
            obj.params.outputTrain = outputTrain;
            outputLength = size(obj.params.outputTrain,1);

            problem.nvars = outputLength;
            problem.Aeq = obj.params.outputTrain';
            problem.beq = 0;
            problem.lb = ones(1,outputLength)*0;
            problem.ub = ones(1,outputLength)*obj.params.boxConstraints;
            problem.fitnessfcn = @(individual) obj.Fitness(individual,inputTrain,outputTrain,obj.params.kernelParams);
            problem.options = gaoptimset('Generations',obj.params.generations,...
                                         'Display','iter',...
                                         'StallGenLimit',200);
            [individual] = ga(problem);
            
            obj.params.supportVectors = obj.params.inputTrain(individual > 1e-9,:);
            obj.params.supportVectorsLabels = obj.params.outputTrain(individual> 1e-9,:);
            obj.params.bias = 0;
            obj.params.alpha = individual(individual > 1e-9);
        end
        function outputhat = predict(obj,input)
            kernelValues = obj.calcKernel(obj.params.supportVectors',input',...
                obj.params.kernelParams);
            outputhat = sign(obj.params.bias + sum(repmat(obj.params.supportVectorsLabels' .*...
                obj.params.alpha,[size(input,1) 1]) .* kernelValues,2));
        end
    end
    methods(Static)
        function out = calcKernel(X1, X2, params)
            switch params.type
                case 'linear'
                    out = X2'*X1;
                case 'poly'
                    out = (X2'*X1 + 1).^params.degree;
                otherwise
                    out = X2'*X1;
            end
        end
        function out = Fitness(individual,inputTrain,outputTrain,params)
            A = sum(individual);
            ki = ESVM.calcKernel(inputTrain',inputTrain',params);
            B = (individual' * individual) .* (outputTrain * outputTrain') .* ki;
            out = -(1/2)*sum(B(:) + A);
        end
    end
end

