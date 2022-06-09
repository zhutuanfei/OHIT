function Syn=OHIT(data,label,curMinLabel,eta,k,kapa,drT)
%---------------OHIT: oversampling for time series data--------------------
% data: the sample set
% label: the class label
% curMinLabel: the label value of minority class
% eta: the number of synthetic samples required to be created
% k,kapa,drT: three parameters in DRSNN

min_ind = find(label==curMinLabel);
n = numel(min_ind);
data = data(min_ind,:); % the minority sample set
m = size(data,2);
% --------------set paramters for OHIT
if nargin<4 || nargin>7
    error('erroneous parmater number!');
end
if nargin==4
k=ceil(n^(1/2)*1.25);kapa=ceil(n^(1/2));drT=0.9;
end
if nargin==5
kapa=ceil(n^(1/2));drT=0.9;
end
if nargin==6
drT=0.9;
end

%---------------deal with exception
if eta<=0
    Syn = [];
    return;
end
if n==1
   Syn = repmat(data,eta,1); 
   return; 
end
Syn = zeros(eta,m);

%------------performing DRSNN clustering
[clusters,clusterLabel]=DRSNN(data, k, kapa, drT);
%no cluster has been found, the whole samples are taken as one cluster
if isempty(clusters)  
    clusters{1}=1:size(data,1);
    clusterLabel = ones(size(data,1),1);
end

%-------------compute the shrinkage covariance
%-------for each cluster
Me = zeros(numel(clusters),m);
eigenMatrix = cell(numel(clusters),1);
eigenValue = cell(numel(clusters),1);
for i=1:numel(clusters)
    [Me(i,:), eigenMatrix{i}, eigenValue{i}] = covStruct(data(clusters{i},:));
end

%---------------allocate the number of synthetic
%-------samples for each cluster 
os_ind = [];
os_ind =[os_ind;repmat((1:n)',floor(eta/n),1)];
os_ind = [os_ind;randsample((1:n)',eta-n*floor(eta/n),false)];
if numel(clusters)>1
   R=1.25; 
else
   R = 1.1;
end

%----------------generate  the structure-preserving 
%--------synthetic samples for each cluster
count = 0;
for i = 1:numel(clusters)
    gen_i = sum(ismember(os_ind,find(clusterLabel==i)));
    Syn(count+1:count+gen_i,:) = OverSamp(Me(i,:), eigenMatrix{i}, eigenValue{i}, gen_i, R);
    os_ind = os_ind(~ismember(os_ind,clusters{i}));
    count = count + gen_i;
end

end

function SampGen = OverSamp(Me, eigenMatrix, eigenValue, eta, R)
cnt = 0;
SampGen = zeros(round(eta*R), length(Me));
Prob    = zeros(round(eta*R),1);
DD = sqrt(abs(diag(eigenValue)));
DD = reshape(DD,1,numel(DD));
Mu = zeros(1,numel(Me));
Sigma = eye(numel(Me));
while cnt < R*eta
    cnt = cnt + 1;
    S = mvnrnd(Mu, Sigma, 1);
    Prob(cnt) = mvnpdf(S, Mu, Sigma);
    S = S.*DD;
    x = S*eigenMatrix'+ Me;
    SampGen(cnt,:) = x;
%     Prob(cnt,1)    = tp; 
end
[~,ind]=sort(Prob,'descend');
SampGen = SampGen(ind(1:eta),:);
end


function [Me, eigenMatrix, eigenValue] = covStruct(P)
sigma=shrinkageCov(P);
Me = mean(P, 1); 
[eigenMatrix,eigenValue]=eigs(sigma,size(P,2));
eigenValue = abs(eigenValue);
end

function [clusters,clusterLabel]=DRSNN(data,k,kapa,drT)
% DRSNN: a density-ratio based shared nearest neighbor clustering
%----------------------description of parameters--------------------------- 
% data: the minority sample set; row--object/sample; column--attribute.
% k: the nearest neighbor parameter in SNN similarity
% kapa: the nearest neighbor parameter in defining density ratio
% drT: the density-ratio threshold
%-----------------------------------------------------------------------end
if nargin ~=4
   error('erroneous parmater number!')
end
n=size(data,1);

%--------------find the k-nearest neighbors for each sample 
%--------------according to certain direct distance metric
IDX=knnsearch(data, data, 'K', k+1, 'Distance', 'euclidean');
eIDX=[];
for i=1:n  
    it_index=find(IDX(i,1:k+1)==i, 1);
    if ~isempty(it_index)
        IDX(i,it_index)=-1;
    else
        IDX(i,k+1)=-1;
    end
    eIDX(i,:)=IDX(i,IDX(i,:)~=-1);
end

%-------------compute SNN  similarity
strength=zeros(n,n);
for i=1:n          
    nni=eIDX(i,:);
for j=(i+1):n
	nnj=eIDX(j,:);
    sharednn = intersect(nni,nnj);
    for l = 1:numel(sharednn)  %SNN  similarity with weight version
       strength(i,j) = (k+1-find(nni==sharednn(l)))*(k+1-find(nnj==sharednn(l)))+strength(i,j); 
    end
    strength(j,i)=strength(i,j);
end
end

%-------------construct shared nearest neighbor graph
[strengthNN,IDXNN]=sort(strength,2,'descend');
strengthNN = strengthNN(:,1:k);
graph = zeros(n,k);  
for i=1:n
    for j=1:k
    if ~isempty(find(IDXNN(IDXNN(i,j),1:k)==i, 1)) 
        graph(i,j)=1;
    end
    end
end

%-------------compute density for each sample
density = sum(strengthNN.*graph,2); 
                                 
%-------------compute density ratio for each sample 
density_ratio = zeros(n,1);
for i = 1:n   %eliminate the negative effect from noisy samples
    non_noise = find(density(IDXNN(i,1:kapa))~=0);  
    if isempty(non_noise)
       density_ratio(i)=0;
    else
       density_ratio(i)=density(i)/(mean(density(IDXNN(i,non_noise))));
    end
end

%-------------identify core points
core_ind = find(density_ratio>=drT);

%-------------find directly density-reachable samples for each core point
neigbhorhood = cell(numel(core_ind),1);
for i = 1:numel(core_ind)
   neigbhorhood{i}= IDXNN(core_ind(i),1:kapa);
   for j = 1:numel(core_ind)
       if sum(IDXNN(core_ind(j),1:kapa)==core_ind(i))~=0
           neigbhorhood{i} = [neigbhorhood{i} core_ind(j)];
       end
   end
end

%-------------build the clusters
clusterLabel = zeros(n,1);
[clusters, clusterLabel] = expandCluster(core_ind, neigbhorhood, clusterLabel);

% noise = find(clusterLabel==0);
% for i=1:numel(noise)
%     max_num=-1;
%     for j=1:numel(clusters)
%         num = sum(clusterLabel(IDXNN(noise(i),1:kapa))==j);
%         if num>max_num   %how many neighbors are from j-th cluster
%             max_num = num;
%             clusterLabel(noise(i))=j;
%         end
%     end
% end
end

function [clusters, clusterLabel] = expandCluster(core_ind, neigbhorhood, clusterLabel) 
clusters = [];
id = 0;
for i=1:numel(core_ind)
    if clusterLabel(core_ind(i))==0
       id = id +1;
       seed = core_ind(i);
       clusters{id}=seed;
       while ~isempty(seed)
           ind = find(core_ind==seed(1), 1);
           if ~isempty(ind) && clusterLabel(seed(1))==0
               seed = [seed neigbhorhood{ind}];
               clusters{id}=union(clusters{id},neigbhorhood{ind}); %border points can be added into multiple clusters, rather only one
           end
           clusterLabel(seed(1))=id;
           seed(1)=[];
       end 
    end
end
end

function [sigma,shrinkage]=shrinkageCov(x)
% shrinkageCov: shrinkage estimate for Cov matrix
%----------------------description of parameters---------------------------
% x: the sample data

t=size(x,1);
n=size(x,2);
meanx=mean(x);
x=x-meanx(ones(t,1),:);
xmkt=mean(x')';

sample=cov([x xmkt])*(t-1)/t;
covmkt=sample(1:n,n+1);  % the vector of slope estimates
varmkt=sample(n+1,n+1);  % the variance of market returns
sample(:,n+1)=[];  
sample(n+1,:)=[];
prior=covmkt*covmkt'./varmkt;
prior(logical(eye(n)))=diag(sample); %F = s_{00}^2 bb' + D

%------------compute the shrinkage intensity
c=norm(sample-prior,'fro')^2;
y=x.^2;
p=1/t*sum(sum(y'*y))-sum(sum(sample.^2));
  % r is divided into diagonal
  % and off-diagonal terms, and the off-diagonal term
  % is itself divided into smaller terms 
rdiag=1/t*sum(sum(y.^2))-sum(diag(sample).^2);
z=x.*xmkt(:,ones(1,n));
v1=1/t*y'*z-covmkt(:,ones(1,n)).*sample;
roff1=sum(sum(v1.*covmkt(:,ones(1,n))'))/varmkt...
	  -sum(diag(v1).*covmkt)/varmkt;
v3=1/t*z'*z-varmkt*sample;
roff3=sum(sum(v3.*(covmkt*covmkt')))/varmkt^2 ...
	  -sum(diag(v3).*covmkt.^2)/varmkt^2;
roff=2*roff1-roff3;
r=rdiag+roff;
k=(p-r)/c;
shrinkage=max(0,min(1,k/t))

%------------compute the estimator
sigma=shrinkage*prior+(1-shrinkage)*sample;

% This file is released under the BSD 2-clause license.
% Copyright (c) 2014, Olivier Ledoit and Michael Wolf 
% All rights reserved.
end
