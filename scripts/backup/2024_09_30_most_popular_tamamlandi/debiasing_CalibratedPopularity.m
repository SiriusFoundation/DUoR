function [TopNRecs] = debiasing_CalibratedPopularity (Dataset,PredMatrix,CandidateListSize, NSize)
% Input:
% Dataset: mxn format rating dataset
% Predictions: nx3 prediction data
% Output:
% TopNRecs: mx10 format topn list where m is the number of users
% CandidateListSize = 100;

%% Construct Prediction Matrix (mxn)
Predictions=zeros(size(Dataset,1),size(Dataset,2));
for row=1:size(PredMatrix,1)
	UserID=PredMatrix(row,1);
	ItemID=PredMatrix(row,2);
	Rating=PredMatrix(row,3);
	Predictions(UserID,ItemID)=Rating;
end



%% Determine short, mid and head item set:
Pop=sum(Dataset~=0);
PopItems=Pop/size(Dataset,1);
[outPop,idxPop]=sort(Pop,'descend');
LimitHead=(sum(Pop))*20/100;
LimitMid=(sum(Pop))*60/100;
top=0;HeadIDX=[]; TailIDX=[]; MidIDX=[];

for i=1:size(idxPop,2)
    if (top<=LimitHead)
        top=top+outPop(1,i);
        HeadIDX = [HeadIDX; idxPop(1,i)];
    elseif (top>LimitHead && top<=LimitMid)
        top=top+outPop(1,i);
        MidIDX = [MidIDX; idxPop(1,i)];
    else
        top=top+outPop(1,i);
        TailIDX = [TailIDX; idxPop(1,i)];
    end
end

TopNRecs = zeros(size(Dataset,1),NSize);
for user=1:size(Dataset,1)
    % user
    TopNRecs(user,:)=reRanking(Dataset(user,:), Predictions(user,:), TailIDX, HeadIDX, MidIDX, CandidateListSize, NSize);
end



return
end

function [TopN] = reRanking (Profile, Predictions, TailSet, HeadSet, MidSet, CandidateListSize, N)
%%  Input parameters:
%   Profile:    user rating profile 1xn
%   Predictions:prediction vectors for the user 1xn
%   TailSet:    Set of tail items nx1
%   HeadSet:    Set of head items nx1
%   N: size of top-N list
lambda = 0.9;

%% Calculate Distribution of mid, head, and tail items in the user profile
[a,RatedItems]=find(Profile(1,:)~=0);
TailCount=0; HeadCount=0; MidCount=0;
for item=1:size(RatedItems,2)
    itemID=RatedItems(1,item);
    if(ismember(itemID,HeadSet))
        HeadCount=HeadCount+1;
    end
    if(ismember(itemID,MidSet))
    MidCount=MidCount+1;
    end
    if(ismember(itemID,TailSet))
    TailCount=TailCount+1;
    end
end

% Construct userVector for Jensen-Shannon Divergence
profileSize=size(RatedItems,2);
userVector = [HeadCount/profileSize MidCount/profileSize TailCount/profileSize];

Preds=normalize(Predictions(1,:),'range');  % normalize predictions into [0,1] scale
[ListPreds, ListIDx] = maxk(Preds(1,:),CandidateListSize); % Determine Initial list with 100 items
TopN = zeros(1,N);  % Final top-N list

for count=1:N  % Construct recommendation loop
    if(count==1)
        TopN(1,count)=ListIDx(1,1);
        ListIDx(:,1)=[];
        ListPreds(:,1)=[];
    else
        
        % head, mid, and tail ratios of current Top-N list
   
        TailCount=0; HeadCount=0; MidCount=0;
        [a,TopNItems]=find(TopN(1,:)~=0);
        for i=1:size(TopNItems,2)
            itemID=TopN(1,i);
            if(ismember(itemID,HeadSet))
                HeadCount=HeadCount+1;
            end
            if(ismember(itemID,MidSet))
                MidCount=MidCount+1;
            end
            if(ismember(itemID,TailSet))
                TailCount=TailCount+1;
            end
        end
        RecSize=size(TopNItems,2)+1;

       % RecommendationVector = [HeadCount/RecSize MidCount/RecSize TailCount/RecSize];

        % Compute ranking scores
        RankingScores=[];
        for i=1:size(ListIDx,2)
            RecommendationVector =[];
            ItemID=ListIDx(1,i);
            if(ismember(ItemID, HeadSet))
                RecommendationVector = [(HeadCount+1)/RecSize MidCount/RecSize TailCount/RecSize];
            end
            if(ismember(ItemID, TailSet))
                RecommendationVector = [HeadCount/RecSize MidCount/RecSize (TailCount+1)/RecSize];
            end
            if(ismember(ItemID,MidSet))
                RecommendationVector = [HeadCount/RecSize (MidCount+1)/RecSize TailCount/RecSize];
            end
            RankingScores(1,i) = ((1-lambda)*ListPreds(1,i)) - (lambda*JSDiv(RecommendationVector,userVector));
        end

        [value,itemCol] = max(RankingScores);
        ItemID=ListIDx(1,itemCol);
        TopN(1,count)=ItemID;
        ListIDx(:,itemCol)=[];
        ListPreds(:,itemCol)=[];
    end
end


return
end


%% Jensen-Shannon divergence


function dist=JSDiv(P,Q)
% Jensen-Shannon divergence of two probability distributions
%  dist = JSD(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
% have the length of one at each 
% P =  n x nbins
% Q =  1 x nbins
% dist = n x 1


if size(P,2)~=size(Q,2)
    error('the number of columns in P and Q should be the same');
end

% normalizing the P and Q
Q = Q ./sum(Q);
Q = repmat(Q,[size(P,1) 1]);
P = P ./repmat(sum(P,2),[1 size(P,2)]);

M = 0.5.*(P + Q);

dist = 0.5.*KLDiv(P,M) + 0.5*KLDiv(Q,M);

return
end


function dist=KLDiv(P,Q)
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
% have the length of one at each 
% P =  n x nbins
% Q =  1 x nbins or n x nbins(one to one)
% dist = n x 1



if size(P,2)~=size(Q,2)
    error('the number of columns in P and Q should be the same');
end

if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
   error('the inputs contain non-finite values!') 
end

% normalizing the P and Q
if size(Q,1)==1
    Q = Q ./sum(Q);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    dist =  sum(P.*log(P./repmat(Q,[size(P,1) 1])),2);
    
elseif size(Q,1)==size(P,1)
    
    Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    dist =  sum(P.*log(P./Q),2);
end

% resolving the case when P(i)==0
dist(isnan(dist))=0;


return
end








