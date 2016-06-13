classdef classifier < handle
    %CLASSIFIER é a classe padrão para criação classificadores
    
    properties
        params % Struct com todos os parâmatros do classificador
    end
    
    methods
        function obj = classifier(params)
            obj.params = params;
        end
        function errorValue = error(obj, inputTest, outputTest)
            outputHat = obj.predict(inputTest);
            confusionMatrix = confusionmat(outputTest, outputHat);
            errorValue = 1 - (trace(confusionMatrix) / sum(confusionMatrix(:)));
        end
        function plot(obj, varargin)
            if nargin < 2
                divisions = 100;
            else
                divisions = varargin{1};
            end
            if size(obj.params.inputTrain,2) ~= 2
                fprintf('Input training have mor tath two dimensions\n');
            else
                xInitial = min(obj.params.inputTrain(:,1));
                xFinal   = max(obj.params.inputTrain(:,1));
                yInitial = min(obj.params.inputTrain(:,2));
                yFinal   = max(obj.params.inputTrain(:,2));
                
                xStep    = (xFinal - xInitial)/divisions;
                yStep    = (yFinal - yInitial)/divisions;
                
                [xGrid,yGrid] = meshgrid(xInitial-(50*xStep):xStep:xFinal+(50*xStep),...
                                         yInitial-(50*yStep):yStep:yFinal+(50*yStep));
                imageSize = size(xGrid);
                xyGrid = [xGrid(:) yGrid(:)];
                
                outputHat = obj.predict(xyGrid);
                
                decisionMap = reshape(outputHat, imageSize);
                
                f = figure('Units','inches',...
                      'Position',[2 2 5 4],...
                      'PaperPositionMode','auto');
                set(f,'defaulttextinterpreter','latex');
                hold on;
                marker = {'r+','g*','bs','yd'};
                
                uniqueLabels = unique(outputHat);
                
                
                if size(obj.params.outputTrain,2) > 1
                    [~,obj.params.outputTrain] = max(obj.params.outputTrain,[],2);
                end
                
                for i=1:max(uniqueLabels)
                    inputSamples{i} = obj.params.inputTrain(logical(obj.params.outputTrain==uniqueLabels(i)),:);
                    %scatter(cl2{i}(:,1),cl2{i}(:,2),30, marker(i,:),'filled');
                    plot(inputSamples{i}(:,1),inputSamples{i}(:,2),marker{i});
                end
                box on;
                set(gca,'ydir','normal');
                contour(xGrid, yGrid, decisionMap, 'Color', 'k','LineWidth',.01);
                hold off;
            end
        end
    end
end

