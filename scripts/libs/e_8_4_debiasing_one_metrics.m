clc;
clear;

disp("Start e_9_one_metrics");

% veri setleri



m_dataset_name = "MLM";
m_dataset_name = "DoubanBooks";
m_dataset_name = "Yelp";

% path belirlendi
m_path = strcat('../out/8_3_debiasing_evaluations_2/',m_dataset_name,'_all.csv');
m_save_path = strcat('../out/8_4_debiasing_one_metric/', m_dataset_name,'_debiasing_one_metric_all.xlsx');

% veri okunuyor
disp("Start reading");
m_Data_Set = readtable(m_path);

% okunan tablo temizleniyor
% kolonlar rename yapıldı
m_Data_Set = renamevars(m_Data_Set,["NCDG","Novelity"],["NDCG","Novelty"]);

% gereksiz kolonlar silindi
m_Data_Set = removevars(m_Data_Set,{'GAPp'});
m_Data_Set = removevars(m_Data_Set,{'GAPr'});
m_Data_Set = removevars(m_Data_Set,{'DeltaGAP_IM'});
m_Data_Set = removevars(m_Data_Set,{'DeltaGAP_MA'});
m_Data_Set = removevars(m_Data_Set,{'date'});
m_Data_Set = removevars(m_Data_Set,{'counter'});
m_Data_Set = removevars(m_Data_Set,{'filename'});
m_Data_Set = removevars(m_Data_Set,{'randomize'});

% orijinal verinin değerleri alındı ve onun satırı da silindi
m_org_row = m_Data_Set(strcmp(m_Data_Set.calculation, 'org'), :);
org_ndcg = m_org_row.NDCG;
org_precision = m_org_row.Precision;
org_recall = m_org_row.Recall;
org_f1 = m_org_row.F1;
org_aplt = m_org_row.APLT;
org_novelty = m_org_row.Novelty;
org_ltc = m_org_row.LTC;
org_entropy = m_org_row.Entropy;

m_Data_Set(ismember(m_Data_Set.algorithm,'org'),:)=[];
m_Data_Set = removevars(m_Data_Set,{'prediction'});

disp("cleaned");


% orijinal degerlerin atanması için kolonlar kopyalanıyor
m_Data_Set.org_NDCG = (m_Data_Set.NDCG);
m_Data_Set.org_Precision = (m_Data_Set.Precision);
m_Data_Set.org_Recall = (m_Data_Set.Recall);
m_Data_Set.org_F1 = (m_Data_Set.F1);

m_Data_Set.org_APLT = (m_Data_Set.APLT);
m_Data_Set.org_Novelty = (m_Data_Set.Novelty);
m_Data_Set.org_LTC = (m_Data_Set.LTC);
m_Data_Set.org_Entropy = (m_Data_Set.Entropy);

% orijinal degerler atanıyor
for i=1:height(m_Data_Set)
    m_Data_Set.org_NDCG(i) = org_ndcg;
    m_Data_Set.org_Precision(i) = org_precision;
    m_Data_Set.org_Recall(i) = org_recall;
    m_Data_Set.org_F1(i) = org_f1;

    m_Data_Set.org_APLT(i) = org_aplt;
    m_Data_Set.org_Novelty(i) = org_novelty;
    m_Data_Set.org_LTC(i) = org_ltc;
    m_Data_Set.org_Entropy(i) = org_entropy;
end

% yeni harmonik ortalama kolonu eklendi
m_Data_Set.harmmean5 = (m_Data_Set.NDCG);

% yeni harmonik ortalama kolonu dupdate edildi
for i=1:height(m_Data_Set)
    m_harm_mean_vector = [m_Data_Set.NDCG(i), m_Data_Set.APLT(i), m_Data_Set.Novelty(i), m_Data_Set.LTC(i), m_Data_Set.Entropy(i)];
    m_harm_mean = harmmean(m_harm_mean_vector);
    m_Data_Set.harmmean5(i) = m_harm_mean;
end

% yeni pair kolonları ve ortalamalarının kolonu eklendi
m_Data_Set.pair_APLT = (m_Data_Set.APLT);
m_Data_Set.pair_Novelty = (m_Data_Set.Novelty);
m_Data_Set.pair_LTC = (m_Data_Set.LTC);
m_Data_Set.pair_Entropy = (m_Data_Set.Entropy);
m_Data_Set.pair_Average = ((m_Data_Set.APLT + m_Data_Set.Novelty + m_Data_Set.LTC + m_Data_Set.Entropy) / 4);

% yeni pair kolonları ve ortalamalarının kolonu guncellendi
for i=1:height(m_Data_Set)
    m_harm_mean_pair_vector = [m_Data_Set.NDCG(i), m_Data_Set.APLT(i)];
    m_harm_mean_pair = harmmean(m_harm_mean_pair_vector);
    m_Data_Set.pair_APLT(i) = m_harm_mean_pair;

    m_harm_mean_pair_vector = [m_Data_Set.NDCG(i), m_Data_Set.Novelty(i)];
    m_harm_mean_pair = harmmean(m_harm_mean_pair_vector);
    m_Data_Set.pair_Novelty(i) = m_harm_mean_pair;

    m_harm_mean_pair_vector = [m_Data_Set.NDCG(i), m_Data_Set.LTC(i)];
    m_harm_mean_pair = harmmean(m_harm_mean_pair_vector);
    m_Data_Set.pair_LTC(i) = m_harm_mean_pair;

    m_harm_mean_pair_vector = [m_Data_Set.NDCG(i), m_Data_Set.Entropy(i)];
    m_harm_mean_pair = harmmean(m_harm_mean_pair_vector);
    m_Data_Set.pair_Entropy(i) = m_harm_mean_pair;

    m_Data_Set.pair_Average(i) = (m_Data_Set.pair_APLT(i) + m_Data_Set.pair_Novelty(i) + m_Data_Set.pair_LTC(i) + m_Data_Set.pair_Entropy(i)) / 4;
end

% şimdi de aynı 6 kolon orijinal veri için yapılacak

m_Data_Set.org_harmmean5 = (m_Data_Set.org_NDCG);

for i=1:height(m_Data_Set)
    m_harm_mean_vector_org = [m_Data_Set.org_NDCG(i), m_Data_Set.org_APLT(i), m_Data_Set.org_Novelty(i), m_Data_Set.org_LTC(i), m_Data_Set.org_Entropy(i)];
    m_harm_mean_org = harmmean(m_harm_mean_vector_org);
    m_Data_Set.org_harmmean5(i) = m_harm_mean_org;
end

m_Data_Set.pair_org_APLT = (m_Data_Set.org_APLT);
m_Data_Set.pair_org_Novelty = (m_Data_Set.org_Novelty);
m_Data_Set.pair_org_LTC = (m_Data_Set.org_LTC);
m_Data_Set.pair_org_Entropy = (m_Data_Set.org_Entropy);
m_Data_Set.pair_org_Average = ((m_Data_Set.org_APLT + m_Data_Set.org_Novelty + m_Data_Set.org_LTC + m_Data_Set.org_Entropy) / 4);

for i=1:height(m_Data_Set)
    m_harm_mean_pair_vector_org = [m_Data_Set.org_NDCG(i), m_Data_Set.org_APLT(i)];
    m_harm_mean_pair_org = harmmean(m_harm_mean_pair_vector_org);
    m_Data_Set.pair_org_APLT(i) = m_harm_mean_pair_org;

    m_harm_mean_pair_vector_org = [m_Data_Set.org_NDCG(i), m_Data_Set.org_Novelty(i)];
    m_harm_mean_pair_org = harmmean(m_harm_mean_pair_vector_org);
    m_Data_Set.pair_org_Novelty(i) = m_harm_mean_pair_org;

    m_harm_mean_pair_vector_org = [m_Data_Set.org_NDCG(i), m_Data_Set.org_LTC(i)];
    m_harm_mean_pair_org = harmmean(m_harm_mean_pair_vector_org);
    m_Data_Set.pair_org_LTC(i) = m_harm_mean_pair_org;

    m_harm_mean_pair_vector_org = [m_Data_Set.org_NDCG(i), m_Data_Set.org_Entropy(i)];
    m_harm_mean_pair_org = harmmean(m_harm_mean_pair_vector_org);
    m_Data_Set.pair_org_Entropy(i) = m_harm_mean_pair_org;

    m_Data_Set.pair_org_Average(i) = (m_Data_Set.pair_org_APLT(i) + m_Data_Set.pair_org_Novelty(i) + m_Data_Set.pair_org_LTC(i) + m_Data_Set.pair_org_Entropy(i)) / 4;
end

% şimdi de iyileşmeler
m_Data_Set.imp_NDCG = m_Data_Set.NDCG;
m_Data_Set.imp_Precision = m_Data_Set.Precision;
m_Data_Set.imp_Recall = m_Data_Set.Recall;
m_Data_Set.imp_F1 = m_Data_Set.F1;

m_Data_Set.imp_APLT = m_Data_Set.APLT;
m_Data_Set.imp_Novelty = m_Data_Set.Novelty;
m_Data_Set.imp_LTC = m_Data_Set.LTC;
m_Data_Set.imp_Entropy = m_Data_Set.Entropy;

m_Data_Set.imp_Harmmean5 = m_Data_Set.harmmean5;

m_Data_Set.imp_pair_APLT = m_Data_Set.pair_APLT;
m_Data_Set.imp_pair_Novelty = m_Data_Set.pair_Novelty;
m_Data_Set.imp_pair_LTC = m_Data_Set.pair_LTC;
m_Data_Set.imp_pair_Entropy = m_Data_Set.pair_Entropy;

m_Data_Set.imp_pair_Average = m_Data_Set.pair_Average;

for i=1:height(m_Data_Set)

    m_Data_Set.imp_NDCG(i) = (100 * m_Data_Set.NDCG(i) / m_Data_Set.org_NDCG(i)) - 100;
    m_Data_Set.imp_Precision(i) = (100 * m_Data_Set.Precision(i) / m_Data_Set.org_Precision(i)) - 100;
    m_Data_Set.imp_Recall(i) = (100 * m_Data_Set.Recall(i) / m_Data_Set.org_Recall(i)) - 100;
    m_Data_Set.imp_F1(i) = (100 * m_Data_Set.F1(i) / m_Data_Set.org_F1(i)) - 100;

    m_Data_Set.imp_APLT(i) = (100 * m_Data_Set.APLT(i) / m_Data_Set.org_APLT(i)) - 100;
    m_Data_Set.imp_Novelty(i) = (100 * m_Data_Set.Novelty(i) / m_Data_Set.org_Novelty(i)) - 100;
    m_Data_Set.imp_LTC(i) = (100 * m_Data_Set.LTC(i) / m_Data_Set.org_LTC(i)) - 100;
    m_Data_Set.imp_Entropy(i) = (100 * m_Data_Set.Entropy(i) / m_Data_Set.org_Entropy(i)) - 100;

    m_Data_Set.imp_Harmmean5(i) = (100 * m_Data_Set.harmmean5(i) / m_Data_Set.org_harmmean5(i)) - 100;

    m_Data_Set.imp_pair_APLT(i) = (100 * m_Data_Set.pair_APLT(i) / m_Data_Set.pair_org_APLT(i)) - 100;
    m_Data_Set.imp_pair_Novelty(i) = (100 * m_Data_Set.pair_Novelty(i) / m_Data_Set.pair_org_Novelty(i)) - 100;
    m_Data_Set.imp_pair_LTC(i) = (100 * m_Data_Set.pair_LTC(i) / m_Data_Set.pair_org_LTC(i)) - 100;
    m_Data_Set.imp_pair_Entropy(i) = (100 * m_Data_Set.pair_Entropy(i) / m_Data_Set.pair_org_Entropy(i)) - 100;

    m_Data_Set.imp_pair_Average(i) = (100 * m_Data_Set.pair_Average(i) / m_Data_Set.pair_org_Average(i)) - 100;
end

disp(strcat(m_dataset_name, " done"));

writetable(m_Data_Set,m_save_path,'FileType','spreadsheet','Sheet',m_dataset_name);

disp("Finish e_9_one_metrics");
