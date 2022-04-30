function newdata = OHIT(data,label)
% 继承自OTIHcm31，不对噪声过采样，
% 噪声不纳入聚类去估计方差结构
% 不进行特征值和特征向量求值

oversampleSize = computeNumberOfPatternstoOversample(label);

if sum(oversampleSize)==0  %if the size to oversampling is zero, no oversampling is needed
    newdata = [label data];
%     newlabel = label;
    return;
end

newdata = [];
% newlabel = zeros(sum(oversampleSize),1);
classes = unique(label);
count=0;
for i=1:numel(oversampleSize)
    if oversampleSize(i)>0  
       SI=doOTIH(data,label,classes(i),oversampleSize(i));
       newdata(count+1:count+size(SI,1),:)= [ones(size(SI,1),1).*classes(i) SI];
       count = count + size(SI,1);
    end
end

newdata = [newdata;[label data]];
end

function SI=doOTIH(data,label,curMinLabel,gen_n)
min_ind = find(label==curMinLabel); %the indexes of current data class
% other_ind = find(label~=curMinLabel);
% np = numel(min_ind);
np = numel(min_ind);
% k1=round(np*2/3);k2=7;k3=7;drThre = 0.9;
% k1=ceil(np^(1/2))+1;k2=ceil(np^(1/2));k3=ceil(np^(1/2));drThre=0.9;
k1=ceil(np^(1/2)*1.25);k2=ceil(np^(1/2));k3=ceil(np^(1/2));drThre=0.9;

% rTh=min((n/(n+nn)),1/2); %锟斤拷cluster锟斤拷锟斤拷目没锟斤拷锟斤拷锟斤拷一锟斤拷锟斤拷锟斤拷
data = data(min_ind,:);
m = size(data,2);

%----------------------------------deal with exception
if gen_n<=0
    SI = [];
    return;
end

if np==1
   SI = repmat(data,gen_n,1); 
   return; 
end

SI = zeros(gen_n,m);
 
%------------------------performing the NBDOS clustering algorithm
clusters=DRSNN(data, k1, k2,k3, drThre);

% clusters = LSNNC(data, k1, k2, thred, k3);
% [clus_label,clId]  = NBDOS(nn_ind(:,1:k2),min_ind,k2,rTh); %
if isempty(clusters)  %if there is no cluster in feature space, the whole samples 
    % are taken into account
    clusters{1}=1:size(data,1);
%     clusterLabel = ones(size(data,1),1);
end
% compute covariance for each outstanding cluster
Me = zeros(numel(clusters),m);
eigenMatrix = cell(numel(clusters),1);
eigenValue = cell(numel(clusters),1);

for i=1:numel(clusters)
       [Me(i,:), eigenMatrix{i}, eigenValue{i}] = covStruct(data(clusters{i},:));
end
                            
% os_ind = [];
% os_ind =[os_ind;repmat((1:np)',floor(gen_n/np),1)];
% os_ind = [os_ind;randsample((1:np)',gen_n-np*floor(gen_n/np),false)];

if numel(clusters)>1
   R=1.25; 
else
   R = 1.1;
end

allmembers=0;
for i=1:numel(clusters)
    allmembers=numel(clusters{i})+allmembers;
end
count = 0;
for i = 1:numel(clusters)
%     gen_i = sum(ismember(os_ind,find(clusterLabel==i)));
    gen_i = round(gen_n*(numel(clusters{i})/allmembers));
    SI(count+1:count+gen_i,:) = OverSamp(Me(i,:), eigenMatrix{i}, eigenValue{i}, gen_i, R);
    count = count + gen_i;
end

% count = 0;
% for i = 1:numel(clusters)
%     gen_i = sum(ismember(os_ind,find(clusterLabel==i)));
%     SI(count+1:count+gen_i,:) = OverSamp(Me(i,:), eigenMatrix{i}, eigenValue{i}, gen_i, R);
%     os_ind = os_ind(~ismember(os_ind,clusters{i}));
%     count = count + gen_i;
% end

%------------------------oversampling outliers or noise with identity guass
%distribution
% noise = find(clusterLabel==0);
% if ~isempty(os_ind) && ~isempty(noise) %noise samples exist 
%     gen_o = numel(os_ind);
%     SI(count+1:count+gen_o,:) = OverNoise(data(noise,:),nearCluster,eigenMatrix, eigenValue,gen_o, R,beta);
% end
end

function SampGen = OverSamp(Me, eigenMatrix, eigenValue, gen_n, R)
% R = 1; %the generated instances are selected from how much times synthetic instances
cnt = 0;

SampGen = zeros(round(gen_n*R), length(Me));
Prob    = zeros(round(gen_n*R),1);
DD = sqrt(abs(diag(eigenValue)));
DD = reshape(DD,1,numel(DD));

Mu = zeros(1,numel(Me));
Sigma = eye(numel(Me));
while cnt < R*gen_n
    cnt = cnt + 1;

    S = mvnrnd(Mu, Sigma, 1);
    Prob(cnt) = mvnpdf(S, Mu, Sigma);
    S = S.*DD;
    
    
    x = S*eigenMatrix'+ Me;
    SampGen(cnt,:) = x;
%     Prob(cnt,1)    = tp; 
end

[~,ind]=sort(Prob,'descend');
SampGen = SampGen(ind(1:gen_n),:);
end


function [Me, eigenMatrix, eigenValue] = covStruct(P)
sigma=covMarket(P);
Me = mean(P, 1); %compute mean for each dimnesion
% P = P - repmat(Me,size(P,1),1); %make samples to have zero mean 
[eigenMatrix,eigenValue]=eigs(sigma,size(P,2));
eigenValue = abs(eigenValue);
end

function [clusters,clusterLabel]=DRSNN(data, k1, k2,k3, drThre,distance)
% a density ratio-based shared nearest neighbor clustering
% compute simialr matrix according to certain direct distance metric
% compute simialr matrix according to SNN with tak1ing the ordering of the
% near neighbors into account
% construct shared nearest neighbor graph
% compute density for each sample
% compute density ratio for each sample
% determind core samples
% determind connected neighborhood

% 
% data input. row: object/sample; column: attribute.
% k1: nearest neighbor
% distance: methods for calculation distance (euclidean, correlation ...)
if nargin <5
   error('not enough arguments given: data, output file name.')
end
switch nargin
	case 5
		distance='euclidean';
end

n=size(data,1);
%%%------compute simialr matrix according to certain direct distance metric
IDX=knnsearch(data, data, 'K', k1+1, 'Distance', distance);
eIDX=[];
for i=1:n  %remove the itself index 
    it_index=find(IDX(i,1:k1+1)==i, 1);
    if ~isempty(it_index)
        IDX(i,it_index)=-1;
    else
        IDX(i,k1+1)=-1;
    end
    eIDX(i,:)=IDX(i,IDX(i,:)~=-1);
end

%----compute simialr matrix according to SNN with tak1ing the ordering into
% account
strength=zeros(n,n);
for i=1:n          %compute the similar matrix
    nni=eIDX(i,:);
for j=(i+1):n
	nnj=eIDX(j,:);
    sharednn = intersect(nni,nnj);
    for l = 1:numel(sharednn)  %shared neighbor neighbor simalir with weight version
       strength(i,j) = (k1+1-find(nni==sharednn(l)))*(k1+1-find(nnj==sharednn(l)))+strength(i,j); 
    end
% 	strength(i,j)=numel(intersect(nni,nnj));
    strength(j,i)=strength(i,j);
    % the closeness depend on the rank1 of the shared k1nn in both list
%     if ~isempty(find(nni==j, 1))&&~isempty(find(nnj==i, 1))
%         graph(i,j)=1;
%         graph(j,i)=1;
%     end
end
end

%---------------------construct shared nearest neighbor graph
[strengthNN,IDXNN]=sort(strength,2,'descend');
strengthNN = strengthNN(:,1:k1);
% IDXNN = IDXNN(:,1:k1);
graph = zeros(n,k1);   %adjenct graph
for i=1:n
    for j=1:k1
    if ~isempty(find(IDXNN(IDXNN(i,j),1:k1)==i, 1)) % bulid the graph for k1-nearest neighbors with each other
        graph(i,j)=1;
    end
    end
end

%---------------------compute density for each sample
density = sum(strengthNN.*graph,2);  % obtain the density for each sample
% neigbhorhood = IDXNN.*graph;               % e-neighborhood for each sample %why reversay nearest neighbors with core are not added into this 
                                    %neighborhood, since some hubs are actually close to points in different clusters

                                    
%---------------------compute density ratio for each sample 
density_ratio = zeros(n,1);
for i = 1:n
    non_noise = find(density(IDXNN(i,1:k2))~=0);  %exclude out the effect from noise samples
    if isempty(non_noise)
       density_ratio(i)=0;
    else
       density_ratio(i)=density(i)/(mean(density(IDXNN(i,non_noise))));
    end
%     if density(i)~=0 && mean(density(IDXNN(i,:)))
%        density_ratio(i)=density(i)/(mean(density(IDXNN(i,:))));
%     end
end

%-------------determind core points
core_ind = find(density_ratio>=drThre);
%-------------find connected neighborhood for each sample
neigbhorhood = cell(numel(core_ind),1);
for i = 1:numel(core_ind)
   neigbhorhood{i}= IDXNN(core_ind(i),1:k3);
   for j = 1:numel(core_ind)
       if sum(IDXNN(core_ind(j),1:k3)==core_ind(i))~=0
           neigbhorhood{i} = [neigbhorhood{i} core_ind(j)];
       end
   end
end
%------------expand clusters
clusterLabel = zeros(n,1);
[clusters, clusterLabel] = expandCluster(core_ind, neigbhorhood, clusterLabel);

%------------extended cluster
% density_cluster = zeros(numel(clusters,1));
% noise_ind = find(clusterLabel==0);
% for i=1:numel(clusters)
%     min_similar = 100000;
%     [core_ind_i,core_ind_ind_i] = intersect(core_ind,clusters{i}); %the indexs of core points belongging to cluster{i}
%     for j=1:numel(core_ind_i)
%         directly_reachable_core=intersect(core_ind_i,neigbhorhood{core_ind_ind_i(j)});
%         min_similar_j = min(strength(core_ind_i(j),directly_reachable_core));%the maximum similar of the directly density-reachable dimilar among core samples 
%         if min_similar_j < min_similar
%            min_similar = min_similar_j;
%            density_cluster(i) = min_similar;
%         end
%     end
%     for j=1:numel(noise_ind)
%         max_similar_j = max(strength(noise_ind(j),core_ind_i));
%         if max_similar_j >= density_cluster(i)
%            clusters{i} = union(clusters{i}, noise_ind(j));
%         end
%     end
% end
% 
% noise = find(clusterLabel==0);
% nearCluster = zeros(numel(noise),1); %find the nearest cluster for each noise sample
% for i=1:numel(noise)
%     max_num=-1;
%     for j=1:numel(clusters)
%         num = sum(clusterLabel(IDXNN(noise(i),1:k3))==j);
%         if num>max_num   %how many neighbors are from j-th cluster
%             nearCluster(i)=j;
%             max_num = num;
%             clusterLabel(noise(i))=j;
%         end
%     end
% end
% nearCluster=[];
% cluster_id = unique(clusterLabel);
% for i = 1: numel(cluster_id)
%     clusters{i}=find(clusterLabel==cluster_id);
% end
end

function [clusters, clusterLabel] = expandCluster(core_ind, neigbhorhood, clusterLabel) 
clusters = [];
% core_ind = find(density_ratio>=drThre); % the indexes of core pointes
id = 0;
for i=1:numel(core_ind)
    if clusterLabel(core_ind(i))==0
       id = id +1;
       seed = core_ind(i);
       clusters{id}=seed;
       while ~isempty(seed)
           ind = find(core_ind==seed(1), 1);
           if ~isempty(ind) && clusterLabel(seed(1))==0%if the considered point belongs to core point
%                add = neigbhorhood(seed(1),neigbhorhood(seed(1),:)~=0); %the e-neighborhood of seed(1) is added into the considered seed
               seed = [seed neigbhorhood{ind}];
               clusters{id}=union(clusters{id},neigbhorhood{ind}); %border points can be added into different clusters, rather only one
           end
           clusterLabel(seed(1))=id;
           seed(1)=[];
       end 
    end
end
end
