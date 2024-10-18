clc;
clear;

disp("Start c9_plot");

% pathler belirlendi
m_ImportPath = "../out/8_rafined_evaluations/";

%active_dataSet = "MLM";
active_dataSet = "Yelp";
%active_dataSet = "DoubanBooks";

m_ExportPath = "../out/9_plot_prepare/" + active_dataSet + "/";
m_algorithm_vector = ["HPF", "IA", "IBPR", "MMMF", "MP", "NEUMF", "SKM", "VAECF", "WBPR", "WMF"];
m_algorithm_vector_ordered = ["MP", "IA", "MMMF", "WMF", "HPF", "IBPR", "WBPR", "SKM", "NEUMF", "VAECF"];
m_profile_vector = ["non_privacy", "fiery_fundamentalist", "pretty_pragmatist", "marginally_concerned"];
m_profile_vector_ordered = ["non_privacy", "marginally_concerned", "pretty_pragmatist", "fiery_fundamentalist"];


% profiles eklendi
fiery_fundamentalist_sigma = 4;
fiery_fundamentalist_beta = 25;
pretty_pragmatist_sigma = 3; 
pretty_pragmatist_beta = 10;
marginally_concerned_sigma = 2;
marginally_concerned_beta = 5;
non_privacy_sigma = 0;
non_privacy_beta = 0;

% veri okuma formatı
opts = delimitedTextImportOptions("NumVariables", 21);
opts.DataLines = [2, Inf];
opts.Delimiter = ";";
opts.VariableNames = ["date", "counter", "filename", "dataset", "sigma", "beta", "dist", "randomize", "algorithm", "GAPp", "GAPr", "DeltaGAP_IM", "DeltaGAP_MA", "NCDG", "Precision", "Recall", "F1", "APLT", "Novelity", "LTC", "Entropy"];
opts.VariableTypes = ["datetime", "double", "string", "categorical", "double", "double", "categorical", "double", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "filename", "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["filename", "dataset", "dist", "algorithm"], "EmptyFieldRule", "auto");

% veri okundu
m_DataSetFull = readtable(strcat(m_ImportPath, active_dataSet,".csv"), opts);

% veri kümesi profillerle daraltıldı
% dağılım olarak N seçildi
m_DataSet_SubSet_as_DT = m_DataSetFull( ...
    (m_DataSetFull.sigma == fiery_fundamentalist_sigma & m_DataSetFull.beta == fiery_fundamentalist_beta & m_DataSetFull.dist == "N") | ...
    (m_DataSetFull.sigma == pretty_pragmatist_sigma & m_DataSetFull.beta == pretty_pragmatist_beta & m_DataSetFull.dist == "N") | ...
    (m_DataSetFull.sigma == marginally_concerned_sigma & m_DataSetFull.beta == marginally_concerned_beta & m_DataSetFull.dist == "N") | ...
    (m_DataSetFull.sigma == non_privacy_sigma & m_DataSetFull.beta == non_privacy_beta & m_DataSetFull.dist == "N"),:);

% gereksiz kolonlar revmoe edildi
m_DataSet_SubSet_as_DT = removevars(m_DataSet_SubSet_as_DT,{'date','counter', 'dataset','randomize'});

% filename kolonu profilename'e cevrildi
m_DataSet_SubSet_as_DT.Properties.VariableNames = ["profile", "sigma", "beta", "dist", "algorithm", "GAPp", "GAPr", "DeltaGAP_IM", "DeltaGAP_MA", "NCDG", "Precision", "Recall", "F1", "APLT", "Novelity", "LTC", "Entropy"];

% döngü için cell e convert edildi
m_DataSet_SubSet_as_Cell = table2cell(m_DataSet_SubSet_as_DT);

% profile name kolonu sigma ve betaya gore guncellendi
for dataset_Counter=1:size(m_DataSet_SubSet_as_Cell,1)
    if (m_DataSet_SubSet_as_Cell{dataset_Counter,2} == fiery_fundamentalist_sigma & m_DataSet_SubSet_as_Cell{dataset_Counter,3} == fiery_fundamentalist_beta)
        m_DataSet_SubSet_as_Cell{dataset_Counter,1} = 'fiery_fundamentalist';
    elseif (m_DataSet_SubSet_as_Cell{dataset_Counter,2} == pretty_pragmatist_sigma & m_DataSet_SubSet_as_Cell{dataset_Counter,3} == pretty_pragmatist_beta)
        m_DataSet_SubSet_as_Cell{dataset_Counter,1} = 'pretty_pragmatist';
    elseif (m_DataSet_SubSet_as_Cell{dataset_Counter,2} == marginally_concerned_sigma & m_DataSet_SubSet_as_Cell{dataset_Counter,3} == marginally_concerned_beta)
        m_DataSet_SubSet_as_Cell{dataset_Counter,1} = 'marginally_concerned';
    elseif (m_DataSet_SubSet_as_Cell{dataset_Counter,2} == non_privacy_sigma & m_DataSet_SubSet_as_Cell{dataset_Counter,3} == non_privacy_beta)
        m_DataSet_SubSet_as_Cell{dataset_Counter,1} = 'non_privacy';
    end
end

% sonuc tablosu olusturuldu, bazı kolonlar remove edildi
m_results_table = cell2table(m_DataSet_SubSet_as_Cell);
m_results_table.Properties.VariableNames = ["profile", "sigma", "beta", "dist", "algorithm", "GAPp", "GAPr", "DeltaGAP_IM", "DeltaGAP_MA", "NCDG", "Precision", "Recall", "F1", "APLT", "Novelity", "LTC", "Entropy"];
m_results_table = removevars(m_results_table,{'sigma','beta', 'dist'});


% ------------------------ metrikler alınıyor -------------------------

% GAPp 
m_results_table_GAPp = m_results_table;
m_results_table_GAPp = removevars(m_results_table_GAPp,{'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_GAPp = table2cell(m_results_table_GAPp);
m_results_vector_GAPp = zeros(10,4,'double');

for m_results_table_GAPp_counter=1:size(m_results_table_GAPp,1)
    m_active_profile = m_results_table_GAPp{m_results_table_GAPp_counter,1};
    m_active_algorithm = string(m_results_table_GAPp{m_results_table_GAPp_counter,2});
    m_active_value = m_results_table_GAPp{m_results_table_GAPp_counter,3};
    m_results_vector_GAPp(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_GAPp = [m_profile_vector_ordered; num2cell(m_results_vector_GAPp)];
writematrix(m_results_vector_GAPp, strcat(m_ExportPath, "01_GAPp.csv"), 'Delimiter', ';');

% GAPr 
m_results_table_GAPr = m_results_table;
m_results_table_GAPr = removevars(m_results_table_GAPr,{'GAPp', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_GAPr = table2cell(m_results_table_GAPr);
m_results_vector_GAPr = zeros(10,4,'double');


for m_results_table_GAPr_counter=1:size(m_results_table_GAPr,1)
    m_active_profile = m_results_table_GAPr{m_results_table_GAPr_counter,1};
    m_active_algorithm = string(m_results_table_GAPr{m_results_table_GAPr_counter,2});
    m_active_value = m_results_table_GAPr{m_results_table_GAPr_counter,3};
    m_results_vector_GAPr(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_GAPr = [m_profile_vector_ordered; num2cell(m_results_vector_GAPr)];
writematrix(m_results_vector_GAPr, strcat(m_ExportPath, "02_GAPr.csv"), 'Delimiter', ';');



% GAPr - percentage - MERT
% GAPr 
m_results_table_GAPr_percentage = m_results_table;
m_results_table_GAPr_percentage = removevars(m_results_table_GAPr_percentage,{'GAPp', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_GAPr_percentage = table2cell(m_results_table_GAPr_percentage);
m_results_vector_GAPr_percentage = zeros(10,4,'double');


for m_results_table_GAPr_percentage_counter=1:size(m_results_table_GAPr_percentage,1)
    m_active_profile = m_results_table_GAPr_percentage{m_results_table_GAPr_percentage_counter,1};
    m_active_algorithm = string(m_results_table_GAPr_percentage{m_results_table_GAPr_percentage_counter,2});
    m_active_value = m_results_table_GAPr_percentage{m_results_table_GAPr_percentage_counter,3};
    m_results_vector_GAPr_percentage(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_GAPr_percentage = [m_profile_vector_ordered; num2cell(m_results_vector_GAPr_percentage)];

% yazmadan önce bu aşamada oranlar hesaplanacak

% (np - mc) / mc * 100

% hata ve fark var
for m_results_table_GAPr_percentage_counter=2:size(m_results_vector_GAPr_percentage,1)
    m_np_value = m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 1);
    m_mc_value = m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 2);
    m_pp_value = m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 3);
    m_ff_value = m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 4);

    
    new_mc = (str2num(m_np_value) - str2num(m_mc_value)) / str2num(m_mc_value) * 100;
    new_pp = (str2num(m_np_value) - str2num(m_pp_value)) / str2num(m_pp_value) * 100;
    new_ff = (str2num(m_np_value) - str2num(m_ff_value)) / str2num(m_ff_value) * 100;
    m_np_value = (str2num(m_np_value) - str2num(m_np_value)) / str2num(m_mc_value) * 100;

    m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 1) = m_np_value; 
    m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 2) = new_mc;
    m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 3) = new_pp;
    m_results_vector_GAPr_percentage(m_results_table_GAPr_percentage_counter, 4) = new_ff;
end


disp("done");

writematrix(m_results_vector_GAPr_percentage, strcat(m_ExportPath, "02_GAPr_percentage.csv"), 'Delimiter', ';');




% DeltaGAP_IM 
m_results_table_DeltaGAP_IM = m_results_table;
m_results_table_DeltaGAP_IM = removevars(m_results_table_DeltaGAP_IM,{'GAPp', 'GAPr',        'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_DeltaGAP_IM = table2cell(m_results_table_DeltaGAP_IM);
m_results_vector_DeltaGAP_IM = zeros(10,4,'double');

for m_results_table_DeltaGAP_IM_counter=1:size(m_results_table_DeltaGAP_IM,1)
    m_active_profile = m_results_table_DeltaGAP_IM{m_results_table_DeltaGAP_IM_counter,1};
    m_active_algorithm = string(m_results_table_DeltaGAP_IM{m_results_table_DeltaGAP_IM_counter,2});
    m_active_value = m_results_table_DeltaGAP_IM{m_results_table_DeltaGAP_IM_counter,3};
    m_results_vector_DeltaGAP_IM(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_DeltaGAP_IM = [m_profile_vector_ordered; num2cell(m_results_vector_DeltaGAP_IM)];
writematrix(m_results_vector_DeltaGAP_IM, strcat(m_ExportPath, "03_DeltaGAP_IM.csv"), 'Delimiter', ';');


% DeltaGAP_MA 
m_results_table_DeltaGAP_MA = m_results_table;
m_results_table_DeltaGAP_MA = removevars(m_results_table_DeltaGAP_MA,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_DeltaGAP_MA = table2cell(m_results_table_DeltaGAP_MA);
m_results_vector_DeltaGAP_MA = zeros(10,4,'double');

for m_results_table_DeltaGAP_MA_counter=1:size(m_results_table_DeltaGAP_MA,1)
    m_active_profile = m_results_table_DeltaGAP_MA{m_results_table_DeltaGAP_MA_counter,1};
    m_active_algorithm = string(m_results_table_DeltaGAP_MA{m_results_table_DeltaGAP_MA_counter,2});
    m_active_value = m_results_table_DeltaGAP_MA{m_results_table_DeltaGAP_MA_counter,3};
    m_results_vector_DeltaGAP_MA(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_DeltaGAP_MA = [m_profile_vector_ordered; num2cell(m_results_vector_DeltaGAP_MA)];
writematrix(m_results_vector_DeltaGAP_MA, strcat(m_ExportPath, "04_DeltaGAP_MA.csv"), 'Delimiter', ';');


% NCDG - % 'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'
% 'profile', 'algorithm', 'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'

m_results_table_NDCG = m_results_table;
m_results_table_NDCG = removevars(m_results_table_NDCG,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_NDCG = table2cell(m_results_table_NDCG);
m_results_vector_NDCG = zeros(10,4,'double');

for m_results_table_NDCG_counter=1:size(m_results_table_NDCG,1)
    m_active_profile = m_results_table_NDCG{m_results_table_NDCG_counter,1};
    m_active_algorithm = string(m_results_table_NDCG{m_results_table_NDCG_counter,2});
    m_active_value = m_results_table_NDCG{m_results_table_NDCG_counter,3};
    m_results_vector_NDCG(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_NDCG = [m_profile_vector_ordered; num2cell(m_results_vector_NDCG)];
writematrix(m_results_vector_NDCG, strcat(m_ExportPath, "05_NDCG.csv"), 'Delimiter', ';');



% Precision 
m_results_table_Precision = m_results_table;
m_results_table_Precision = removevars(m_results_table_Precision,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_Precision = table2cell(m_results_table_Precision);
m_results_vector_Precision = zeros(10,4,'double');

for m_results_table_Precision_counter=1:size(m_results_table_Precision,1)
    m_active_profile = m_results_table_Precision{m_results_table_Precision_counter,1};
    m_active_algorithm = string(m_results_table_Precision{m_results_table_Precision_counter,2});
    m_active_value = m_results_table_Precision{m_results_table_Precision_counter,3};
    m_results_vector_Precision(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_Precision = [m_profile_vector_ordered; num2cell(m_results_vector_Precision)];
writematrix(m_results_vector_Precision, strcat(m_ExportPath, "06_Precision.csv"), 'Delimiter', ';');


% Recall 
m_results_table_Recall = m_results_table;
m_results_table_Recall = removevars(m_results_table_Recall,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_Recall = table2cell(m_results_table_Recall);
m_results_vector_Recall = zeros(10,4,'double');

for m_results_table_Recall_counter=1:size(m_results_table_Recall,1)
    m_active_profile = m_results_table_Recall{m_results_table_Recall_counter,1};
    m_active_algorithm = string(m_results_table_Recall{m_results_table_Recall_counter,2});
    m_active_value = m_results_table_Recall{m_results_table_Recall_counter,3};
    m_results_vector_Recall(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_Recall = [m_profile_vector_ordered; num2cell(m_results_vector_Recall)];
writematrix(m_results_vector_Recall, strcat(m_ExportPath, "07_Recall.csv"), 'Delimiter', ';');


% F1 
m_results_table_F1 = m_results_table;
m_results_table_F1 = removevars(m_results_table_F1,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'APLT', 'Novelity', 'LTC', 'Entropy'});
m_results_table_F1 = table2cell(m_results_table_F1);
m_results_vector_F1 = zeros(10,4,'double');

for m_results_table_F1_counter=1:size(m_results_table_F1,1)
    m_active_profile = m_results_table_F1{m_results_table_F1_counter,1};
    m_active_algorithm = string(m_results_table_F1{m_results_table_F1_counter,2});
    m_active_value = m_results_table_F1{m_results_table_F1_counter,3};
    m_results_vector_F1(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_F1 = [m_profile_vector_ordered; num2cell(m_results_vector_F1)];
writematrix(m_results_vector_F1, strcat(m_ExportPath, "08_F1.csv"), 'Delimiter', ';');


% APLT 
m_results_table_APLT = m_results_table;
m_results_table_APLT = removevars(m_results_table_APLT,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'Novelity', 'LTC', 'Entropy'});
m_results_table_APLT = table2cell(m_results_table_APLT);
m_results_vector_APLT = zeros(10,4,'double');

for m_results_table_APLT_counter=1:size(m_results_table_APLT,1)
    m_active_profile = m_results_table_APLT{m_results_table_APLT_counter,1};
    m_active_algorithm = string(m_results_table_APLT{m_results_table_APLT_counter,2});
    m_active_value = m_results_table_APLT{m_results_table_APLT_counter,3};
    m_results_vector_APLT(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_APLT = [m_profile_vector_ordered; num2cell(m_results_vector_APLT)];
writematrix(m_results_vector_APLT, strcat(m_ExportPath, "09_APLT.csv"), 'Delimiter', ';');



% Novelity 
m_results_table_Novelity = m_results_table;
m_results_table_Novelity = removevars(m_results_table_Novelity,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'LTC', 'Entropy'});
m_results_table_Novelity = table2cell(m_results_table_Novelity);
m_results_vector_Novelity = zeros(10,4,'double');

for m_results_table_Novelity_counter=1:size(m_results_table_Novelity,1)
    m_active_profile = m_results_table_Novelity{m_results_table_Novelity_counter,1};
    m_active_algorithm = string(m_results_table_Novelity{m_results_table_Novelity_counter,2});
    m_active_value = m_results_table_Novelity{m_results_table_Novelity_counter,3};
    m_results_vector_Novelity(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_Novelity = [m_profile_vector_ordered; num2cell(m_results_vector_Novelity)];
writematrix(m_results_vector_Novelity, strcat(m_ExportPath, "10_Novelity.csv"), 'Delimiter', ';');



% LTC 
m_results_table_LTC = m_results_table;
m_results_table_LTC = removevars(m_results_table_LTC,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'Entropy'});
m_results_table_LTC = table2cell(m_results_table_LTC);
m_results_vector_LTC = zeros(10,4,'double');

for m_results_table_LTC_counter=1:size(m_results_table_LTC,1)
    m_active_profile = m_results_table_LTC{m_results_table_LTC_counter,1};
    m_active_algorithm = string(m_results_table_LTC{m_results_table_LTC_counter,2});
    m_active_value = m_results_table_LTC{m_results_table_LTC_counter,3};
    m_results_vector_LTC(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_LTC = [m_profile_vector_ordered; num2cell(m_results_vector_LTC)];
writematrix(m_results_vector_LTC, strcat(m_ExportPath, "11_LTC.csv"), 'Delimiter', ';');


% Entropy 
m_results_table_Entropy = m_results_table;
m_results_table_Entropy = removevars(m_results_table_Entropy,{'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC'});
m_results_table_Entropy = table2cell(m_results_table_Entropy);
m_results_vector_Entropy = zeros(10,4,'double');

for m_results_table_Entropy_counter=1:size(m_results_table_Entropy,1)
    m_active_profile = m_results_table_Entropy{m_results_table_Entropy_counter,1};
    m_active_algorithm = string(m_results_table_Entropy{m_results_table_Entropy_counter,2});
    m_active_value = m_results_table_Entropy{m_results_table_Entropy_counter,3};
    m_results_vector_Entropy(find(m_algorithm_vector_ordered == m_active_algorithm), find(m_profile_vector_ordered == m_active_profile)) = m_active_value;
end

m_results_vector_Entropy = [m_profile_vector_ordered; num2cell(m_results_vector_Entropy)];
writematrix(m_results_vector_Entropy, strcat(m_ExportPath, "12_Entropy.csv"), 'Delimiter', ';');



clear opts
disp("Finish c9_plot")