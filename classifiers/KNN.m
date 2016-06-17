classdef KNN < Classifier
    %KNN     
    properties
    end
    
    methods
        function obj = KNN(params)
            obj@Classifier(params);
        end
        function obj = train(obj,inputTrain,outputTrain)
            obj.params.inputTrain  = inputTrain;
            obj.params.outputTrain = outputTrain;
        end
        function outputhat = predict(obj,input)
            distances = dist(input, obj.params.inputTrain');
            [~,ordenedOutputs] = sort(distances,2);
            ordenedLabels = obj.params.outputTrain(ordenedOutputs);
            outputhat = mode(ordenedLabels(:,1:obj.params.neighborsNumber),2);
        end
    end
    
end

