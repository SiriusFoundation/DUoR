function [TopNRecs] = debiasing_LNSF(Dataset, PredMatrix, NSize)
% Input:
% Dataset: mxn format rating dataset
% Predictions: nx3 prediction data
% Output:
% TopNRecs: mx10 format topn list where m is the number of users

%% Construct Prediction Matrix (mxn)
Predictions=zeros(size(Dataset,1),size(Dataset,2));
for row=1:size(PredMatrix,1)
	UserID=PredMatrix(row,1);
	ItemID=PredMatrix(row,2);
	Rating=PredMatrix(row,3);
	Predictions(UserID,ItemID)=Rating;
end

% Constructing Q Matrix;
Q_Matrix=zeros(1,size(Dataset,2));
totalNumberRatings=size(nonzeros(Dataset),1);
userSize=size(Dataset,1);
for item=1:size(Dataset,2)
    Q_Matrix(1,item)=size(nonzeros(Dataset(:,item)),1)/totalNumberRatings;
    Q_Matrix(1,item)=Q_Matrix(1,item)*userSize*NSize;
end

% Constructing S Matrix;
S_Matrix=Predictions;
for row=1:size(Predictions,1)
    rowTotal = sum(Predictions(row,:));
    for column=1:size(Predictions,2)
        S_Matrix(row,column)=S_Matrix(row,column)/rowTotal;
    end
end

% Constructing S matrix as rows and sort they in descending order
S_rows = zeros(size(S_Matrix,1)*size(S_Matrix,2),3);
i=1;
for user=1:size(S_Matrix,1)
    %user
    for item=1:size(S_Matrix,2)
        Prediction = S_Matrix(user,item);
        S_rows(i,1)=user;
        S_rows(i,2)=item;
        S_rows(i,3)=Prediction;
        i=i+1;
    end
end
S_rows = sortrows(S_rows,-3);

% Constructing Top-N recommendation lists for all users
TopNRecs = zeros(size(Dataset,1),NSize);

for row=1:size(S_rows,1)
    row;
    if(size(find(TopNRecs==0),1)==0)
        break;
    end
    
    ItemID = S_rows(row,2);
    UserID = S_rows(row,1);
    if(Q_Matrix(1,ItemID)>0 && any(TopNRecs(UserID,:)==0))
        col=find(TopNRecs(UserID,:)==0,1);
        TopNRecs(UserID,col)=ItemID;
        Q_Matrix(1,ItemID)=Q_Matrix(1,ItemID)-1;
    end

end




return
end








