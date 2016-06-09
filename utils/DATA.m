classdef DATA < handle
    %DATA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        path
        name
        dataset
        normaliseType
        normalizedDataset
        outType
        numberOfClasses
        out
    end
    methods
        function obj = DATA(path)
            obj.path    = path;
            diretories = strsplit(path,'/');
            obj.name = diretories{end};
            obj.load();
        end
        function obj = load(obj)
            obj.dataset = dlmread(sprintf('../utils/datasets/%s/data.csv',obj.path));
            obj.dataset = obj.dataset(randperm(size(obj.dataset,1)),:);
            obj.normalizedDataset = obj.dataset;
            obj.numberOfClasses = max(obj.dataset(:,end));
        end
        function obj = normalize(obj,normaliseType)
            obj.normaliseType = normaliseType;
            x = obj.dataset(:,1:end-1);            
            switch obj.normaliseType
                case {'std','standard'}
                    for i=1:size(x,2)
                        media  = mean(x(:,i));
                        desvio = std(x(:,i));
                        obj.normalizedDataset(:,i) = (x(:,i) - media) / desvio;
                    end
                case 'scaling'
                    for i=1:size(x,2)
                        minimo = min(x(:,i));
                        maximo = max(x(:,i));
                        obj.normalizedDataset(:,i) = (x(:,i) - minimo) / (maximo - minimo);
                    end
                otherwise
                    for i=1:size(x,2)
                        minimo = min(x(:,i));
                        maximo = max(x(:,i));
                        obj.normalizedDataset(:,i) = (x(:,i) - minimo) / (maximo - minimo);
                    end
            end
        end
        function obj = divide(obj,varargin)
            if nargin == 2
                if iscell(varargin(1))
                    partition = [.8 .2];
                    obj.outType = num2str(cell2mat(varargin(1)));
                end
            elseif nargin == 1
                obj.outType = 'n';
                partition = [.8 .2];
            else
                partition = cell2mat(varargin(1));
                obj.outType = num2str(cell2mat(varargin(2)));
            end
            count = 0;
            for i=1:length(partition)
                quantity = round(partition(i) * size(obj.normalizedDataset,1));
                obj.out{i}.input   = obj.normalizedDataset(count+1:quantity+count,1:end-1);
                obj.out{i}.output  = obj.normalizedDataset(count+1:quantity+count,end);
                switch obj.outType
                    case {'OneOutOf','oneoutof','oof'}
                        out = zeros(size(obj.out{i}.output,1),max(obj.out{i}.output));
                        for j=1:max(obj.out{i}.output)
                            out(obj.out{i}.output == j,j) = 1;
                        end
                        obj.out{i}.output = out;
                    case {'plusminus','pm'}
                        obj.out{i}.output(obj.out{i}.output == 2) = -1;
                    case {'number','n'}
                    otherwise
                end
                count    = quantity;
            end
        end
    end
end

