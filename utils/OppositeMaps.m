classdef OppositeMaps < handle
    %OPPOSITEMAPS Is a class of the method of reduction set
    properties
        inputSamples            % (In) Matrix of input data
        inputLabels             % (In) Array of labels of input data
        proportion              % (In / Parameter) Proportion of values used in K-means
        
        oppositeMapsIndices     % (Out) Opposite maps indices
        oppositeMapsSamples     % (Out) Opposite maps values
    end
    
    methods
        function obj = OppositeMaps(params)
            obj.proportion = params.proportion;
        end
        function obj = execute(obj, inputSamples, inputLabels)
            obj.inputSamples  = inputSamples;
            obj.inputLabels   = inputLabels;
            % find the instances by class
            positiveIndices = find(inputLabels == +1);
            negativeIndices = find(inputLabels == -1);
            
            positiveSamples = inputSamples(positiveIndices,:);
            negativeSamples = inputSamples(negativeIndices,:);
            positiveNumberOfClusters = ceil(length(positiveIndices) * obj.proportion);
            negativeNumberOfClusters = ceil(length(negativeIndices) * obj.proportion);
            
            [positivePrototypeIndices,positivePrototypes] = kmeans(positiveSamples,positiveNumberOfClusters,'Start','sample','emptyaction','singleton');
            [negativePrototypeIndices,negativePrototypes] = kmeans(negativeSamples,negativeNumberOfClusters,'Start','sample','emptyaction','singleton');

            [~,positiveChampionPrototypes] = min(pdist2(positivePrototypes,negativeSamples));
            positiveChampionPrototypes = unique(positiveChampionPrototypes);
            [~,negativeChampionPrototypes] = min(pdist2(negativePrototypes,positiveSamples));
            negativeChampionPrototypes = unique(negativeChampionPrototypes);
            
            positiveChampions = ismember(positivePrototypeIndices',positiveChampionPrototypes);
            negativeChampions = ismember(negativePrototypeIndices',negativeChampionPrototypes);

            obj.oppositeMapsSamples = [positiveSamples(positiveChampions,:);negativeSamples(negativeChampions,:)];
            obj.oppositeMapsIndices = [positiveIndices(positiveChampions,:);negativeIndices(negativeChampions,:)];
        end
        function plot(obj)
            if size(obj.inputSamples,2) ~= 2
                fprintf('Input training have mor tath two dimensions\n');
            else
                f = figure('Units','inches',...
                          'Position',[2 2 5 4],...
                          'PaperPositionMode','auto');
                    set(f,'defaulttextinterpreter','latex');
                hold on;
                markerSamples = {'r+','g*','bs','yd'};
                markerOppositeMaps = {'bo','ko','bs','yd'};
                classes = [-1 1];
                for i=[1,2]
                    plot(obj.inputSamples(obj.inputLabels == classes(i),1),...
                         obj.inputSamples(obj.inputLabels == classes(i),2),markerSamples{i});
                end

                for i=[1,2]
                    plot(obj.oppositeMapsSamples(obj.inputLabels(obj.oppositeMapsIndices) == classes(i),1),...
                         obj.oppositeMapsSamples(obj.inputLabels(obj.oppositeMapsIndices) == classes(i),2),markerOppositeMaps{i},'MarkerSize',12);
                end
                box on;
                xlimits = [min(obj.inputSamples(:,1)) max(obj.inputSamples(:,1))];
                ylimits = [min(obj.inputSamples(:,2)) max(obj.inputSamples(:,2))];

                xdivision = [-1 1].*(xlimits(2) - xlimits(1))/10;
                ydivision = [-1 1].*(ylimits(2) - ylimits(1))/10;

                xlim(xlimits + xdivision);
                ylim(ylimits + ydivision);
                hold off;
            end
        end
    end
    
end

