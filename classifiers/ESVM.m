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
                                         'CreationFcn',@obj.createIndividuals,...
                                         'MutationFcn',@obj.mutateIndividuals,...
                                         'CrossoverFcn', @obj.crossoverIndividuals,...
                                         'StallGenLimit',200);
            [individual] = ga(problem);
            obj.params.supportVectorsIndices = find(individual > 1e-9);
            
            obj.params.supportVectors = obj.params.inputTrain(obj.params.supportVectorsIndices,:);
            obj.params.supportVectorsLabels = obj.params.outputTrain(obj.params.supportVectorsIndices,:);
            obj.params.bias = 0;
            obj.params.alpha = individual(obj.params.supportVectorsIndices);
        end
        function outputhat = predict(obj,input)
            kernelValues = obj.calcKernel(obj.params.supportVectors',input',...
                obj.params.kernelParams);
            outputhat = sign(obj.params.bias + sum(repmat(obj.params.supportVectorsLabels' .*...
                obj.params.alpha,[size(input,1) 1]) .* kernelValues,2));
        end
        
        function plot(obj, varargin)
            obj.plot@classifier(varargin);
            hold on;
            plot(obj.params.supportVectors(:,1), obj.params.supportVectors(:,2), 'o','MarkerSize', 12);
            legend({'+1', '-1'});
        end
    end
    methods(Static, Hidden)
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
        function population = createIndividuals(GenomeLength,FitnessFcn,options)
            totalPopulation = sum(options.PopulationSize);
            linCon = options.LinearConstr;
            group = linCon.Aeq;
            LB = linCon.lb;
            UB = linCon.ub;
            population = UB(1).*rand(totalPopulation,GenomeLength);
            for i=1:totalPopulation
                population(i,:) = ESVM.ajust(population(i,:),linCon);
            end
        end
        function child = ajust(parent,linCon)
            C = linCon.ub(1);
            group = linCon.Aeq;
            child = parent;
            sumAlphaD = group * child';
            while sumAlphaD >= 1e-9 
                k = randi([1 length(child)]);
                if child(k) > abs(sumAlphaD)
                    child(k) = child(k) - abs(sumAlphaD);
                else
                    child(k) = 0;
                end
                sumAlphaD = group * child';
            end
        end
        function mutationChildren = mutateIndividuals(parents ,options,NVARS, ...
                                    FitnessFcn, state, thisScore,thisPopulation,mutationRate)
            mutationChildren = zeros(length(parents),NVARS);
            linCon = options.LinearConstr;
            group = linCon.Aeq;
            for i=1:length(parents)
                parent = thisPopulation(parents(i),:);
                n = ones(1,2);
                while n(1) == n(2) && group(n(1)) == group(n(2)) && parent(n(1)) == 0 && parent(n(2)) == 0
                    n = randi(length(parent),1,2);
                end
                child = parent;
                child(n(1)) = parent(n(2));
                child(n(2)) = parent(n(1));
                mutationChildren(i,:) = ESVM.ajust(child,linCon);
            end
        end
        function [xoverKids] = crossoverIndividuals(parents,options,GenomeLength,FitnessFcn,unused,thisPopulation)
            nKids = length(parents)/2;
            xoverKids = zeros(nKids,GenomeLength);
            index = 1;
            for i=1:nKids
                 r1 = parents(index);
                 index = index + 1;

                 r2 = parents(index);
                 index = index + 1;

                parentA = thisPopulation(r1,:);
                parentB = thisPopulation(r2,:);

                a = rand;
                c1 = parentA .* a + parentB .* (1 - a);
                c2 = parentB .* a + parentA .* (1 - a);

                if FitnessFcn(c1) > FitnessFcn(c2)
                    xoverKids(i,:) = c2;
                else
                    xoverKids(i,:) = c1;
                end
                xoverKids(i,xoverKids(i,:) < 1e-4) = 0;
            end
        end
    end
end

