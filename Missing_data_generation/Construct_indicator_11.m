
clear;clc;

datasets='WebKB_2views';
% load("UCI_3view.mat")
load('datasets\WebKB_2views.mat')

folds=cell(1,10);

% X{1} = sport01;
% X{2} = sport02;
X=X'
% X= data;
% y= truth;
for i = 1:1
    for ratio=0.1:0.1:0.9
%       for ratio=0.5
        if ~exist(['./Incomplete_index/', datasets])
            mkdir(['./Incomplete_index/', datasets]);
        end      
            for iter=1:10
            % load(datasets);    
            Indicator = BuildIndicator_my(size(X, 2), size(X{1}, 1), ratio);  % number of views , number of samples
            folds{iter}=Indicator;
            end
            save(['./Incomplete_index/',datasets,'/',datasets, 'ratio', int2str(ratio*100),'.mat'], 'folds');
              
    end
end