function  [Avg,Results]=Metrics(Dataset, TopN)
format short g
% ----Input ----
% Dataset: Original data mxn (matris) format
% TopN: topN recommendation lists


%% Construct tail item set
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


%% Calculate metrics for all users:
BTA=zeros(size(Dataset,1),1);
PR=zeros(size(Dataset,1),1);
APRI=zeros(size(Dataset,1),1);
DeviationBTA=zeros(size(Dataset,1),1);
DeviationPR=zeros(size(Dataset,1),1);
NDCG=zeros(size(Dataset,1),1);
Precision=zeros(size(Dataset,1),1);
Recall=zeros(size(Dataset,1),1);
F1=zeros(size(Dataset,1),1);
APLT=zeros(size(Dataset,1),1);
Novelty=zeros(size(Dataset,1),1);
DeltaGAP=zeros(size(Dataset,1),1);


for user=1:size(Dataset,1)
    BTA(user,1)=BTA_Cal(Dataset(user,:),PopItems);
    % PR(user,1)=PR_Cal(Dataset(user,:),PopItems);
    APRI(user,1)=APRI_Cal(TopN(user,:),PopItems);
    % AbsDifferences:
    Abs_BTA(user,1)= (APRI(user,1)-BTA(user,1))^2;
    % Abs_PR(user,1)= abs(APRI(user,1)-PR(user,1));
%     DeviationBTA(user,1)= ((APRI(user,1)-BTA(user,1))/BTA(user,1))*100;
%     if(PR(user,1)~=0)
%     DeviationPR(user,1)= ((APRI(user,1)-PR(user,1))/PR(user,1))*100; 
%     else
%     DeviationPR(user,1)=0;
%     end
    NDCG(user,1)=NDCGFunc(Dataset(user,:),TopN(user,:));
    [Precision(user,1),Recall(user,1),F1(user,1)] = PrecRecallFunc(Dataset(user,:),TopN(user,:));
    APLT(user,1) = APLTFunction(TailIDX,TopN(user,:));
    Novelty(user,1) = NoveltyFunction(Dataset(user,:),TopN(user,:));
    MRM(user,1) = MRMFunction (Dataset(user,:),TopN(user,:), TailIDX, HeadIDX);
end

%% Store all results in Avg variable (1-BTA, 2-APRI, 3-RMSE-PC, 4-NDCG, 5-Precision, 6-Recall, 7-F1, 8-APLT, 9-Novelty, 10-MRM 11-LTC, 12-Entropy)
Results(:,1)=BTA(:,1);
% Results(:,2)=PR(:,1);
Results(:,2)=APRI(:,1);
Results(:,3)=sqrt(sum(Abs_BTA(:,1)) / size(Dataset,1));
% Results(:,5)=Abs_PR(:,1);
% Results(:,6)=DeviationBTA(:,1);
% Results(:,7)=DeviationPR(:,1);
Results(:,4)=NDCG(:,1);
Results(:,5)=Precision(:,1);
Results(:,6)=Recall(:,1);
Results(:,7)=F1(:,1);
Results(:,8)=APLT(:,1);
Results(:,9)=Novelty(:,1);
Results(:,10)=MRM(:,1);
Avg = mean(Results);

Avg(1,11)=LongTailCoverage(TopN, TailIDX);
Avg(1,12)=EntropyCal(Dataset,TopN);

return
end

%% Entropy
function [Entropy] = EntropyCal(Dataset,TopN)

% All recommended items
topnItems=TopN(1,:);
for i=2:size(TopN,1)
    topnItems = cat(2,topnItems,TopN(i,:));
end

itemNumber=size(Dataset,2);
Entropy=0;

for item=1:itemNumber
    pItem=size(find(topnItems==item),2)/size(topnItems,2);
    if(pItem~=0)
    Entropy =  Entropy - pItem*(log(pItem)/log(itemNumber));  
    end
end

return 
end

%% Long-tail coverage
function [LTC] = LongTailCoverage(TopN, TailIDX)

% All recommended items
countLC=0;
topnItems=TopN(1,:);
for i=2:size(TopN,1)
    topnItems = cat(2,topnItems,TopN(i,:));
end
UnItems=unique(topnItems);
for i=1:size(UnItems,2)
        if (any(TailIDX==UnItems(1,i)))
            countLC=countLC+1;
        end
end

LTC=countLC/size(TailIDX,1);


return 
end

%% Better-than-average
function [BTA] = BTA_Cal(Profile, PopItems)
% --- Input ---
% Profile: User profile vector 1xn
% PopItems: Popularity of items 1xn

idx=find(Profile(1,:)~=0); total=0; count=0;
ort = mean(nonzeros(Profile(1,:)));
for i=1:size(idx,2)
    ItemID=idx(1,i);
    if(Profile(1,ItemID)>=ort)
    total=total+PopItems(1,ItemID);
    count=count+1;
    end
end
BTA=total/count;
return
end

%% positively-rated
function [PR] = PR_Cal(Profile, PopItems)
% --- Input ---
% Profile: User profile vector 1xn
% PopItems: Popularity of items 1xn

idx=find(Profile(1,:)~=0); total=0; count=0;
for i=1:size(idx,2)
    ItemID=idx(1,i);
    if(Profile(1,ItemID)>=4)
    total=total+PopItems(1,ItemID);
    count=count+1;
    end
end
if(count~=0)
    PR=total/count;
else
    PR=0;
end

return
end

%% Average Popularity of Recommended Items
function [APRI] = APRI_Cal(TopN, PopItems)
% --- Input ---
% TopN: TopN recommendation list 1xN
% PopItems: Popularity of items nx1

total=0;
for i=1:size(TopN,2)
    ItemID=TopN(1,i);
    total=total+PopItems(1,ItemID);
end
APRI=total/size(TopN,2);
return
end

%% ndcg
function [nDCG] = NDCGFunc(Profile, TopN)
% --- Input ---
% Profile: User profile vector 1xn
% TopN: TopN recommendation list for the user 1xN
DCG=0; IDCG=0;
i=1;
[val,idx]=maxk(Profile(1,:),size(TopN,2));
for item=1:size(TopN,2)
    itemID=TopN(1,item);
    if(i==1)
        DCG=Profile(1,itemID);
        IDCG=val(1,item);
        i=i+1;
    else
        DCG=DCG+(Profile(1,itemID)/log2(item));
        IDCG=IDCG+(val(1,item)/log2(item));
    end
end

nDCG=DCG/IDCG;

return
end

%% precision-recall-f1
function [prec,recall,f1] = PrecRecallFunc(Profile, TopN)
% --- Input ---
% Profile: User profile vector 1xn
% TopN: TopN recommendation list for the user 1xN

count=0;
for i=1:size(TopN,2)
    itemID=TopN(1,i);
    if(Profile(1,itemID)>=4)
        count=count+1;
    end
end

prec=count/size(TopN,2);
AllGood = size(find(Profile(1,:)>=4),2);
if(AllGood==0)
    recall=0;
else
    recall=count/AllGood;
end

if(prec+recall~=0)
    f1=(2*prec*recall)/(prec+recall);
else
    f1=0;
end

return
end

%% APLT
function [aplt] = APLTFunction (TailSet, TopN)
% --- Input ---
% TailSet: The set of tail items nx1
% TopN: TopN recommendation list for the user 1xN
count=0;
for item=1:size(TopN,2)
    itemID=TopN(1,item);
    if(ismember(itemID,TailSet))
        count=count+1;
    end
end

aplt=count/size(TopN,2);

return
end

%% Novelty
function [novelty] = NoveltyFunction(Profile, TopN)
% --- Input ---
% Profile: User profile vector 1xn
% TopN: TopN recommendation list for the user 1xN
count=0;
for item=1:size(TopN,2)
    itemID=TopN(1,item);
    if(Profile(1,itemID)==0)
        count=count+1;
    end
end

novelty = count / size(TopN,2);

return
end

%% MeanRankMiscalibration
function [MRM] = MRMFunction(Profile, TopN, TailIDX, HeadIDX)
% --- Input ---
% Profile: User profile vector 1xn
% TopN: TopN recommendation list for the user 1xN
% TailIDX: Tail item set
% HeadIDX: Head item set

%  MC(p,q):
%  Pay-Fairness(p,q) = İki class için öneri listesiyle kullanıcının gerçek eğilimi arasındaki divergence değeri (Jensen-Shannon)
%  Payda-Fairness (p,q,{}) = q listesi boş olsaydı [0 0] probability için bu değer ne çıkıyor
% 
%  RMC(u):
% 1'den ona kadar kümülatif olarak payın aynı olduğu paydanın ürün boyutuna bağlı olarak
% (1,2,3 ... 10) değiştiği durumlarda divergenceları hesapla ve kümülatif topla!
% Hesaplanan toplamı N değerine böl (10). Sonra bütün kullanıcılar için hesaplayıp kullanıcı sayısına böl
% 
% Not! MC hiç kullanılmayacak aslında 

cumSumMRM=0;
topNCount=0;

for item=1:size(TopN,2)
    TailProfileCount=0; HeadProfileCount=0; TailTopNCount=0; HeadTopNCount=0;
    [valProfile,idxProfiles]=maxk(Profile(1,:),item);
    [idxTopNs]=TopN(:,1:item);
    zerocount=0;
    
    for i=1:item
        if(valProfile(1,i) ~= 0)
            if(any(TailIDX==idxProfiles(1,i)))
            TailProfileCount=TailProfileCount+1;
            else
            HeadProfileCount=HeadProfileCount+1;
            end

            if(any(TailIDX==idxTopNs(1,i)))
            TailTopNCount=TailTopNCount+1;
            else
            HeadTopNCount=HeadTopNCount+1;
            end
            zerocount=zerocount+1; 
        end
    end
    if(i==zerocount)
    ProfileRatio = [HeadProfileCount/zerocount TailProfileCount/zerocount];
    topNRatio = [HeadTopNCount/zerocount TailTopNCount/zerocount]; 
    cumSumMRM = cumSumMRM + JSDiv(ProfileRatio,topNRatio);
    topNCount=topNCount+1;
    end
end
MRM = cumSumMRM/topNCount;

return
end


%% Jensen-Shannon divergence


function jsd = JSDiv(P, Q)
    % Bu fonksiyon P ve Q vektörleri arasındaki Jensen-Shannon Divergence'ı hesaplar.
    % P ve Q: iki olasılık dağılımı vektörü
    % jsd: hesaplanan Jensen-Shannon Divergence değeri

    % Öncelikle, vektörlerin olasılık dağılımı olup olmadığını kontrol edelim
    if sum(P) ~= 1 || sum(Q) ~= 1
        error('P ve Q olasılık dağılımları olmalıdır (toplamları 1 olmalıdır).');
    end

    % Orta nokta dağılımı M'yi hesapla
    M = 0.5 * (P + Q);

    % Kullback-Leibler Divergence fonksiyonunu tanımla
    function kl = kullback_leibler(P, Q)
        % P ve Q'da sıfır olmayan elemanlar için hesaplama yap
        kl = sum(P .* log2(P ./ Q), 'omitnan');
        % Eğer P'de 0 varsa, 0 log(0) = 0 olarak değerlendirilir.
        kl(isnan(kl)) = 0;
    end

    % D_KL(P || M) ve D_KL(Q || M)'yi hesapla
    Dkl_PM = kullback_leibler(P, M);
    Dkl_QM = kullback_leibler(Q, M);

    % Jensen-Shannon Divergence'ı hesapla
    jsd = 0.5 * (Dkl_PM + Dkl_QM);
end

% function dist=JSDiv(P,Q)
% % Jensen-Shannon divergence of two probability distributions
% %  dist = JSD(P,Q) Kullback-Leibler divergence of two discrete probability
% %  distributions
% %  P and Q  are automatically normalised to have the sum of one on rows
% % have the length of one at each 
% % P =  n x nbins
% % Q =  1 x nbins
% % dist = n x 1
% 
% 
% if size(P,2)~=size(Q,2)
%     error('the number of columns in P and Q should be the same');
% end
% 
% % normalizing the P and Q
% Q = Q ./sum(Q);
% Q = repmat(Q,[size(P,1) 1]);
% P = P ./repmat(sum(P,2),[1 size(P,2)]);
% 
% M = 0.5.*(P + Q);
% 
% dist = 0.5.*KLDiv(P,M) + 0.5*KLDiv(Q,M);
% 
% return
% end
% 
% 
% function dist=KLDiv(P,Q)
% %  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
% %  distributions
% %  P and Q  are automatically normalised to have the sum of one on rows
% % have the length of one at each 
% % P =  n x nbins
% % Q =  1 x nbins or n x nbins(one to one)
% % dist = n x 1
% 
% 
% 
% if size(P,2)~=size(Q,2)
%     error('the number of columns in P and Q should be the same');
% end
% 
% if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
%    error('the inputs contain non-finite values!') 
% end
% 
% % normalizing the P and Q
% if size(Q,1)==1
%     Q = Q ./sum(Q);
%     P = P ./repmat(sum(P,2),[1 size(P,2)]);
%     dist =  sum(P.*log(P./repmat(Q,[size(P,1) 1])),2);
%     
% elseif size(Q,1)==size(P,1)
%     
%     Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
%     P = P ./repmat(sum(P,2),[1 size(P,2)]);
%     dist =  sum(P.*log(P./Q),2);
% end
% 
% % resolving the case when P(i)==0
% dist(isnan(dist))=0;
% 
% 
% return
% end











