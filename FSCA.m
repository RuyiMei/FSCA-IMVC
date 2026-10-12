function [yy1,iter_num] = FSCA(X,TIND, ind_0,n, param)
num_view = length(X);

gamma = param.gamma;
beta =  param.beta;
miu = param.miu;
m=param.m;
c= param.c;
ddd = param.ddd;

I = eye(n); 
missingindex = cell(1,num_view);
for iv = 1:num_view
    temp_index = ones(1,n);
    temp_index(1,ind_0{iv}) = 0;
    missingindex{iv} = temp_index;
end

if size(X{1},2)~=n
    for iv = 1:num_view
        X{iv} = X{iv}';
    end
end
for iv = 1:num_view
    X{iv} = NormalizeFea(X{iv},0);
end
clear iv

dims = zeros(1,num_view);
for iv = 1:num_view
    dims(iv) = size(X{iv},1);
end

%% 生成二部图
k=2;
opts.style = 4; % 
%% 锚点采样
BB = cell(num_view,1); % 高阶二部图存储元胞（现在只存一阶）
centers = cell(num_view,1); % 锚点特征矩阵存储元胞
% disp('----------Anchor Selection----------');

if opts.style == 1 % direct sample
    XX = [];
    for v = 1:num_view
       XX = [XX;X{v}];
    end
    [~,ind,~] = graphgen_anchor(XX,m);
    for v = 1:num_view
        centers{v} = X{v}(:, ind);
    end
elseif opts.style == 2 % rand sample
    vec = randperm(n);
    ind = vec(1:m);
    for v = 1:num_view
        centers{v} = X{v}(:, ind);
    end
elseif opts.style == 3 % KNP
    XX = [];
    for v = 1:num_view
        XX = [XX;X{v}];
    end
    [~, ~, ~, ~, dis] = litekmeans(XX', m);
    [~,ind] = min(dis,[],1);
    ind = sort(ind,'ascend');
    for v = 1:num_view
        centers{v} = X{v}(:, ind);  % 索引列，而不是行
    end
elseif opts.style == 4 % kmeans sample
    XX = [];
    for v = 1:num_view
       XX = [XX;X{v}];
       len(v) = size(X{v},1);
    end
    [~, Cen, ~, ~, ~] = litekmeans(XX', m);
    Cen=Cen';
    t1 = 1;
    for v=1:num_view
       t2 = t1+len(v)-1;
       centers{v} = Cen(t1:t2,:);
       t1 = t2+1;
    end
end


%% 各个视角二部图初始化
% disp('---------- Graphs Initialization----------');
% BB = cell(n_view,1); % 每个视角的一阶二部图
W = cell(1,num_view); % 输出二部图转置
for v = 1:num_view
    DD = L2_distance_1(X{v}, centers{v});
    [~, idx] = sort(DD, 2); % sort each row
    BB{v} = zeros(n,m);
    for ii = 1:n
        id = idx(ii,1:k+1);
        di = DD(ii, id);
        BB{v}(ii,id) = (di(k+1)-di) /(k*di(k+1)-sum(di(1:k))+eps);
    end   
end

%% initialization
% initialize W
for v = 1:num_view
    W{v} = BB{v}' * TIND{v};   % m × n_v
end
P = cell(1,v);
% initialize P
for iv = 1:num_view
   di = size(X{iv},1); 
   P{iv} = zeros(di,ddd);
end
A = zeros(ddd,m);
H = zeros(ddd,n);
Z = zeros(m,n); % m  * n
Y = zeros(ddd,n);
% miu = 1e-5; 
rou= 1.5;
max_miu = 10e10;

XXXXX = [];
for p = 1 : num_view
    X{p} = mapstd(X{p},0,1);
    XXXXX = [XXXXX;X{p}];
end
[XU,~,~]=svds(XXXXX',m);
% rand('twister',12);
[IDX,~] = kmeans(XU,m, 'MaxIter',100,'Replicates',10);
for i = 1:n
    Z(IDX(i),i) = 1;
end
%% Training
% MAX_ITER = 15;
MAX_ITER =1000;
tic;
for iter = 1:MAX_ITER 
      H_pre = H;
      Z_pre = Z;
     %% 1.更新P{v}
    for iv = 1:num_view
        M{iv} = X{iv}.*(repmat(missingindex{iv},dims(iv),1))*H';
        [U_b,~,V_b] = svd(M{iv},"econ");
        P{iv} = U_b*V_b';
    end
     %% 2.更新A
     [U,~,V] = svd((H+(Y/miu))*Z','econ');
     A = U*V';
     %% 3.更新H
     sum_H1 = zeros(ddd, n);
     sum_H2 = zeros(n,n);
      for iv = 1:num_view
          H1=P{iv}'*(X{iv}.*(repmat(missingindex{iv},dims(iv),1)));
          % H2=TIND{iv}*(sparse(TIND{iv}'));
          H2=diag(0.5*diag(I).*missingindex{iv}');
          sum_H1=sum_H1+H1;
          sum_H2=sum_H2+H2;
      end
      sum1=2*sum_H1+((miu*A*Z)-Y);
      sum2=2*sum_H2+miu*I;
      H=sum1/ sum2;
     %% 4.更新F 
    % Dm = diag(sum(Z,2).^(-0.5));
    % Dn= diag(sum(Z).^(-0.5)); 
    Dm = diag( (sum(Z,2) + eps).^(-0.5) );
    Dn = diag( (sum(Z)   + eps).^(-0.5) );
    XXXX = Dm*Z*Dn;
    if any(isinf( XXXX), 'all') || any(isnan( XXXX), 'all')
         XXXX(isinf( XXXX)) = 0;
        XXXX(isnan( XXXX)) = 0;
    end
    [U,S,V] = svds( XXXX,c+1);
    Fn = sqrt(2)*U(:,1:c)/2;
    Fm = sqrt(2)*V(:,1:c)/2;
    ev = diag(S);
    Fn_old = Fn;
    Fm_old = Fm;
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 < c-0.0000001
        gamma = 2*gamma;
    elseif fn2 > c+1-0.0000001
        gamma = gamma/2;   Fn = Fn_old; Fm = Fm_old;
    end
    %% 5.更新Z 
    % Compute B
    B = zeros(m,n); 
    for iv = 1:num_view
            B= B + ((beta*(W{iv}* TIND{iv}'))+((A'*Y)/(2*num_view)));
    end
    B = B/((2*beta+miu)/2);
    Fn_ = Dm*Fn;
    Fm_ = Dn*Fm;
    dist = L2_distance_1(Fn_',Fm_');

    Z = zeros(m,n);
    for i = 1:n
        Bi = B(:,i);
        ti = dist(:,i);
        ad = Bi-0.5*gamma*ti;
        Z(:,i) = EProjSimplex_new(ad);
    end
    if any(isnan(Z))
        Z = W{iv}* TIND{iv}';
    end
    
     %% 6.更新Y,miu
     Y=Y+miu*(H-A*Z);
     miu=min(miu*rou,max_miu);
   

    %% 7.记录obj
    tol = 1e-5; 
    
    % for v = 1:num_view
    %     % change_A = change_A + norm(A{v} - A_pre{v}, 'fro');
    % end
    change_H = norm(H - H_pre, 'fro');   
    change_Z = norm(Z - Z_pre, 'fro');
    % change_A = norm(A - A_pre, 'fro');
    if iter > 3
        if change_H < tol && change_Z < tol
            disp('Converged!');
            break;
        end
    end
%   if mod(iter,10)==0
%     fprintf('iter=%d | dH=%.2e dZ=%.2e | r=%.2e | miu=%.2e | gamma=%.2e\n',...
%         iter, change_H, change_Z, norm(H-A*Z,'fro'), miu, gamma);
% end
end
iter_num=iter;
fprintf("iter=%d\n", iter);
%% 9.生成增广图M
G = sparse(n+m,n+m);
G(1:n,n+1:end) = Z';
G(n+1:end,1:n) = Z;
G = (G + G') / 2;  
GGGGG = graph(G);  % 创建图对象，其中 adjacencyMatrix 是你的邻接矩阵
yyy = conncomp(GGGGG);  % bins 返回一个数组，表示每个节点所属的连通组件
yy1 = yyy(1:n)';
end

