clear;
clc;
warning off;
addpath(genpath('./'));
%% ================= Excel 初始化（彻底正确版） =================
excel_name = 'results.xlsx';

% 如果文件已存在，先删除（防止重复 append）
if exist(excel_name,'file')
    delete(excel_name);
end

varNames = { ...
    'miss_rate','m','gamma','beta','miu', ...
    'ACC','NMI','Purity','Fscore','Precision','Recall','AR','Entropy', ...
    'IterNum','Time'};

% 主结果表（只有表头，没有数据）
T_all = table('Size',[0,numel(varNames)], ...
    'VariableTypes', repmat("double",1,numel(varNames)), ...
    'VariableNames', varNames);
writetable(T_all, excel_name);

% Best_ACC 表（只有表头）
T_best = table('Size',[0,numel(varNames)], ...
    'VariableTypes', repmat("double",1,numel(varNames)), ...
    'VariableNames', varNames);
writetable(T_best, excel_name, 'Sheet', 'Best_ACC');

%% ================= Dataset =================
ds = {'WebKB_2views'};

i_fold = 1;

for dsi = 1:length(ds)

    %% ---------- load data ----------
    dataName = ds{dsi};
    load(dataName);
    % Y=y;
    % X=data;
    % Y=truth;
    param.c   = length(unique(Y));
    nCluster  = param.c;
    n         = length(Y);
    num_view  = length(X);

    original_X = X;
    original_Y = Y;

    miss_rates = [0.1 0.3 0.5];

    %% ================= miss_rate 循环 =================
    for i_miss = 1:length(miss_rates)

        % ---------- best 初始化 ----------
        best_ACC   = -inf;
        best_res   = [];
        best_param = struct();
        best_iter  = 0;
        best_time  = 0;

        % ---------- 恢复原始数据 ----------
        X = original_X;
        Y = original_Y;

        mr = miss_rates(i_miss);

        %% ---------- load missing index ----------
        foldName = strcat(dataName,'_percentDel_',num2str(mr),'.mat');
        load(foldName);
        ind_folds = folds{i_fold};

        %% ---------- construct ind_0 ind_1 ----------
        ind_0 = cell(1,num_view);
        ind_1 = cell(1,num_view);
        for iv = 1:num_view            
            ind_0{iv} = find(ind_folds(:,iv) == 0);
            ind_1{iv} = find(ind_folds(:,iv) == 1);
        end
        for v = 1:num_view
            idx = ind_1{v};
            n_v = length(idx);

            Tv = zeros(n, n_v);
             for j = 1:n_v
                   Tv(idx(j), j) = 1;
             end

            TIND{v} = Tv;
        end

        
        %% ---------- initialize X ----------
        for iv = 1:num_view
            if size(X{iv},2) ~= n
                X{iv} = X{iv}';
            end
            X{iv} = NormalizeFea(X{iv}, 0);
            X{iv}(:,ind_0{iv}) = 0;
        end

        %% ---------- parameter setting ----------
        MM = [2*nCluster];
        DDD= [3*nCluster];
        BT = [1e-4 1e-3 1e-2 1e-1 1 10 100];
        CP = [1e-4 1e-3 1e-2 1e-1 1 10 100 1000];
        DT = [1e-4 1e-3 1e-2 1e-1 1 10 100 1000];

        %% ================= grid search =================
        for rp00 = 1:length(DDD)
             param.ddd= DDD(rp00);
        for rp0 = 1:length(MM)
            param.m = MM(rp0);

            for rp2 = 1:length(BT)
                for rp3 = 1:length(CP)
                    for rp4 = 1:length(DT)

                        param.gamma   = BT(rp2);
                        param.beta  = CP(rp3);
                        param.miu = DT(rp4);

                        tic;
                        [yy1, iter_num] = FSCA(X, TIND, ind_0, n, param);
                        time = toc;

                        res = Clustering8Measure(Y, yy1);
                        ACC = res(1);

                        %% ---------- update best ----------
                        if ACC > best_ACC
                            best_ACC   = ACC;
                            best_res   = res;
                            best_param = param;
                            best_iter  = iter_num;
                            best_time  = time;
                        end

                        %% ---------- save all results ----------
                        row_table = table( ...
                            mr, ...
                            param.m, param.gamma, param.beta, param.miu, ...
                            res(1), res(2), res(3), res(4), ...
                            res(5), res(6), res(7), res(8), ...
                            iter_num, time, ...
                            'VariableNames', varNames);

                        writetable(row_table, excel_name, 'WriteMode', 'append');
fprintf(['[mr=%.1f | m=%d | gamma=%.4f | beta=%.4f | miu=%.4f | ' ...
         'iter=%d | time=%.2fs | ACC=%.4f | Best_ACC=%.4f | ' ...
         'c=%d | num_view=%d | n=%d]\n'], ...
         mr, param.m, param.gamma, param.beta, param.miu, ...
         iter_num, time, ACC, best_ACC, ...
         param.c, num_view, n);
                        % 
                        % fprintf('[mr=%.1f | m=%d | gamma=%.2f] ACC=%.4f | Best=%.4f\n', ...
                        %     mr, param.m, param.gamma, ACC, best_ACC);

                    end
                end
            end
        end
     end

        %% ================= 保存当前 miss_rate 的 Best_ACC =================
        best_table = table( ...
            mr, ...
            best_param.m, best_param.gamma, ...
            best_param.beta, best_param.miu, ...
            best_ACC, best_res(2), best_res(3), best_res(4), ...
            best_res(5), best_res(6), best_res(7), best_res(8), ...
            best_iter, best_time, ...
            'VariableNames', varNames);

        writetable(best_table, excel_name, ...
            'Sheet', 'Best_ACC', ...
            'WriteMode', 'append');

        fprintf('>>> BEST (mr=%.1f): ACC=%.4f\n\n', mr, best_ACC);

        clear X Y
    end
end
