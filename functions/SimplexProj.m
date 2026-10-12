% function X = SimplexProj(Y)
% [N,D] = size(Y);
% X = sort(Y,2,'descend');
% Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
% X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
function X = SimplexProj(Y)
    [N, D] = size(Y);
    
    % 对每行降序排序
    X_sorted = sort(Y, 2, 'descend');
    
    % 计算累积和
    cum_X = cumsum(X_sorted, 2) - 1;
    
    % 创建系数矩阵
    if D > 0
        coeff_matrix = diag(sparse(1./(1:D)));
        Xtmp = cum_X * coeff_matrix;
    else
        Xtmp = zeros(N, 0);
    end
    
    % 找到断点位置（处理边界情况）
    mask = X_sorted > Xtmp;
    row_sums = sum(mask, 2);
    row_sums = max(min(row_sums, D), 1);  % 限制在[1, D]范围内
    
    % 获取阈值λ
    linear_indices = sub2ind([N, D], (1:N)', row_sums);
    lambda = Xtmp(linear_indices);
    
    % 计算投影
    X = max(bsxfun(@minus, Y, lambda), 0);
end
