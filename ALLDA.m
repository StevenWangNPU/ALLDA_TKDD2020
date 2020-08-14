% function [W, S] = local_LDA(X, Y, d, h, r, islocal)
function [OBJ,W,S1] = ALLDA(X, Y, d, h, r,perr)
% X: data set, every colmun is a sample
% Y: label vector
% d: final dimension
% h: k nearest neighborhood
% r: paremeter

c = unique(Y);
n = length(Y);
for i=1:length(c)
    Xc{i} = X(:,Y==i);  %  store the each class samples
    nc(i) = size(Xc{i},2);  %  record the number of each class
    ind = find(Y==c(i));
    WW_w(ind,ind) = length(ind)/n;
end
H = eye(n) - 1/n*ones(n);
St = X*H*X';
invSt = inv(St);
pre_S = [];
Obj = 0;
OBJ = [];
interval = 1;
% initial a graph S0 for each class, in which K nearest neighborhood
% of each sample is 1 
for k = 1 : length(c)
    Xi = Xc{k};
    ni = nc(k);   
    distXi = L2_distance_1(Xi,Xi);
    [~, idx] = sort(distXi,2);
    S0{k} = construct_S0( idx, h, ni);   
    pre_S = blkdiag(pre_S,S0{k});      %    
end
% Pre_S is the initial graph

pre_S = pre_S - diag(diag(pre_S));
WW_w = WW_w - diag(diag(WW_w));

% Iterative calculate the projection W with S;   
count  = 0;
while (abs(interval)>=perr&&count<200)
%  for iter = 1:31
 % updata the weight W       
    S1 = [];
    D_w = spdiags(sum(WW_w.*(pre_S),r),0,n,n); % degree matrix
    L_w = D_w - WW_w.*(pre_S);
    L_w = (L_w + L_w')/2;
    Sw = X*L_w*X';
    Sw = (Sw + Sw')/2;
    P = invSt*Sw;  
    [W,~,~] = eig1(P, d, 0, 0);
    W = W*diag(1./sqrt(diag(W'*St*W)));
%   updata the graph S    
    for i=1:length(c)
    Xc{i} = X(:,Y==i); 
    nc(i) = size(Xc{i},2);
    Xi = Xc{i};
    ni = nc(i);
    distXi = L2_distance_1(W'*Xi,W'*Xi);
    [dis, idx] = sort(distXi,2);
    obj(i) = sum((sum (dis(:,2:h+1).^(1/(1-r)),2) ).^(1-r));
    S{i} = construct_S( dis+eps,idx, h, r, ni);
    S1 = blkdiag(S1,S{i});
    end
    % convergence code in next line
    pre_S = S1;
    pre_S = pre_S.^r;
    pre_S = (pre_S+pre_S')/2; 
    interval = Obj - sum(obj);
    OBJ = [OBJ,sum(obj)];
    count = count + 1;  
    Obj =  sum(obj);
end 






