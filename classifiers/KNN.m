classdef KNN < Classifier
    %KNN     
    properties
    end
    
    methods
        function obj = KNN(params)
            obj@Classifier(params);
        end
        function obj = train(obj,inputTrain,outputTrain)
            tic;
            obj.params.inputTrain  = inputTrain;
            
            if size(outputTrain,2) >  1 % Transform labels in 'n'
                [~,outputTrain] = max(outputTrain,[],2);
            end
            obj.metrics.referencePointNumber = size(outputTrain,1);
            obj.params.outputTrain = outputTrain;
            obj.metrics.timeTrain = toc;
        end
        function [outputhat, distances] = predict(obj,input)
            distances = pdist2(input, obj.params.inputTrain);
            [~,ordenedOutputs] = sort(distances,2);
            ordenedLabels = obj.params.outputTrain(ordenedOutputs);
            outputhat = mode(ordenedLabels(:,1:obj.params.K),2);
        end
    end
    
end

