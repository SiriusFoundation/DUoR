function [TopNRecs] = debiasing_BTA(Dataset,PredMatrix,CandidateListSize, NSize)
% Input:
% Dataset: mxn format rating dataset
% Predictions: nx3 prediction data
% Output:
% TopNRecs: mx10 format topn list where m is the number of users
% CandidateListSize

%% Construct Prediction Matrix (mxn)
Predictions=zeros(size(Dataset,1),size(Dataset,2));
for row=1:size(PredMatrix,1)
	UserID=PredMatrix(row,1);
	ItemID=PredMatrix(row,2);
	Rating=PredMatrix(row,3);
	Predictions(UserID,ItemID)=Rating;
end



%% Determine short and head item set:
Pop=sum(Dataset~=0);
PopItems=Pop/size(Dataset,1);
[outPop,idxPop]=sort(Pop,'descend');
LimitHead=(sum(Pop))*20/100;
top=0;HeadIDX=[]; TailIDX=[];
for i=1:size(idxPop,2)
    if (top<=LimitHead)
        top=top+outPop(1,i);
        HeadIDX = [HeadIDX; idxPop(1,i)];
    else
        top=top+outPop(1,i);
        TailIDX = [TailIDX; idxPop(1,i)];
    end
end

TopNRecs = zeros(size(Dataset,1),NSize);
for user=1:size(Dataset,1)
    % user
    TopNRecs(user,:)=reRanking(Dataset(user,:), Predictions(user,:), TailIDX, HeadIDX, CandidateListSize, NSize);
end



return
end

function [TopN] = reRanking (Profile, Predictions, TailSet, HeadSet, CandidateListSize, N)
%%  Input parameters:
%   Profile:    user rating profile 1xn
%   Predictions:prediction vectors for the user 1xn
%   TailSet:    Set of tail items nx1
%   HeadSet:    Set of head items nx1
%   N: size of top-N list
lambda = 0.5;

%% Calculate P(H|u) and P(T|u)
[a,RatedItems]=find(Profile(1,:)~=0);
TailCount=0; HeadCount=0; ProfileCount=0;
RatingAvg=mean(nonzeros(Profile(1,:)));
for item=1:size(RatedItems,2)
    itemID=RatedItems(1,item);
    if(Profile(1,itemID)>=RatingAvg)  % Average üstündeyse dikkate al.
        ProfileCount=ProfileCount+1;
        if(ismember(itemID,HeadSet))
        HeadCount=HeadCount+1;
        end
        if(ismember(itemID,TailSet))
        TailCount=TailCount+1;
        end
    end
end
TailRatio = TailCount/ProfileCount;   % P(T|u)
HeadRatio = HeadCount/ProfileCount;   % P(H|u)

Preds=normalize(Predictions(1,:),'range');  % normalize predictions into [0,1] scale
[ListPreds, ListIDx] = maxk(Preds(1,:),CandidateListSize); % Determine Initial list with specified number of items
TopN = zeros(1,N);  % Final top-N list

for count=1:N  % Construct recommendation loop
    if(count==1)
        TopN(1,count)=ListIDx(1,1);
        ListIDx(:,1)=[];
        ListPreds(:,1)=[];
    else
        % head and tail ratios of current Top-N list
        TailCount=0; HeadCount=0;
        [a,TopNItems]=find(TopN(1,:)~=0);
        for i=1:size(TopNItems,2)
            itemID=TopN(1,i);
            if(ismember(itemID,HeadSet))
                HeadCount=HeadCount+1;
            end
            if(ismember(itemID,TailSet))
                TailCount=TailCount+1;
            end
        end
        TailCoverRatio = TailCount/size(TopNItems,2);  % P(j|c € l) where c=T
        HeadCoverRatio = HeadCount/size(TopNItems,2);% P(j|c € l) where c=H

        % Compute ranking scores
        RankingScores=[];
        for i=1:size(ListIDx,2)
            ItemID=ListIDx(1,i);
            if(ismember(ItemID, HeadSet))
                if(HeadCoverRatio < HeadRatio)
                % RankingScores(1,i) = ((1-lambda)*ListPreds(1,i)) + (lambda*HeadRatio);
                RankingScores(1,i) = ListPreds(1,i);
                else
                RankingScores(1,i) = 0;    
                end
            elseif(ismember(ItemID, TailSet))
                if(TailCoverRatio < TailRatio)
                RankingScores(1,i) = ListPreds(1,i);
                % RankingScores(1,i) = ((1-lambda)*ListPreds(1,i)) + (lambda*TailRatio);
                else
                RankingScores(1,i) = 0;    
                end
            end
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

