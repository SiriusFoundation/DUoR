 n5% 2023_11_02 vesiyonu tamamlandı

% buraya kadar çok temiz gorunuyor, buradan sonrası patlak olabilir
%burası da temiz, sıkıntı göremedim


clc;
clear;

disp("Start e_9_evaluations");

m_dataset = 'DoubanBooks';
m_dataset = 'MLM';
m_dataset = 'Yelp';

data_set_collection = ["Yelp"];

m_path = '../out/1_2_prepare_uir/';
m_raw_dataset_root_path = strcat(m_path,m_dataset,"/");

m_ImportPath = "../out/7_prediction_fake_ratings/";
m_ExportPath = "../out/9_evaluations/";

% data_set_collection = ["MLM", "DoubanBooks", "Yelp", "Dianping"];

% prediction dosyalarında solda indexi olanlar tümü
opts_with_index = delimitedTextImportOptions("NumVariables", 4);
opts_with_index.DataLines = [2, Inf];
opts_with_index.Delimiter = ",";
opts_with_index.VariableNames = ["Var1", "user_id", "item_id", "prediction"];
opts_with_index.SelectedVariableNames = ["user_id", "item_id", "prediction"];
opts_with_index.VariableTypes = ["string", "double", "double", "double"];
opts_with_index.ExtraColumnsRule = "ignore";
opts_with_index.EmptyLineRule = "read";
opts_with_index = setvaropts(opts_with_index, "Var1", "WhitespaceRule", "preserve");
opts_with_index = setvaropts(opts_with_index, "Var1", "EmptyFieldRule", "auto");

% prediction dosyalarında solda indexi olmayanlar
% IA item average ve MP most popular
opts_without_index = delimitedTextImportOptions("NumVariables", 3);
opts_without_index.DataLines = [1, Inf];
opts_without_index.Delimiter = ",";
opts_without_index.VariableNames = ["user_id", "item_id", "prediction"];
opts_without_index.VariableTypes = ["double", "double", "double"];
opts_without_index.ExtraColumnsRule = "ignore";
opts_without_index.EmptyLineRule = "read";

disp("date" + ";" + "counter" + ";" + "filename" + ";" + "dataset" + ";" + "approach" + ";" + "alpha" + ";"  + "randomize" + ";" + "calculation" + ";" + "algorithm" + ";" + "GAPp" + ";" + "GAPr" + ";" + "DeltaGAP_IM" + ";" + "DeltaGAP_MA" + ";" + "NCDG" + ";" + "Precision" + ";" + "Recall" + ";" + "F1" + ";" + "APLT" + ";" + "Novelity" + ";" + "LTC" + ";" + "Entropy");

for dataSetCounter=1:size(data_set_collection,2)
    active_collection = data_set_collection(dataSetCounter);
    tempDirPath = m_ImportPath + active_collection + "/*.csv";
    m_fileList = dir(tempDirPath);
    
    % metriklerin karşılaştırılacağı orijinal veri seti
    raw_dataset_path = m_raw_dataset_root_path + active_collection + '.mat';
    raw_dataset = load(raw_dataset_path);
    raw_temp_dataSet = struct2cell(raw_dataset);
    raw_dataset = raw_temp_dataSet{1};

    % tüm verinin saklanacağı data table oluşturuluyor
    % avg olan 1/5 ölçeğinde, 
    % çünkü 5 randomize örneklerim ortalaması alınacak
    
    % sanırım randomize 1 oldugu için /5 yapmadım
    m_export_cell = cell(size(m_fileList,1),21);
    % m_export_avg_cell = cell((size(m_fileList,1) / 5),21);

    for m_file_counter=1:size(m_fileList,1)
        activeFilename = m_fileList(m_file_counter).name;
        currentPath = m_ImportPath + active_collection + "/" + activeFilename;

        % burada prediction verisi okunuyor
        %if and(m_file_counter >=1, m_file_counter <=2)
        if true
            % eger item averge veya most popular ise soldaki seq kolonu yok, o
            % nedenle farklı formata okuyoruz.
            m_if_fileName = convertCharsToStrings(activeFilename) ;
            if or(m_if_fileName.contains("_MP"), m_if_fileName.contains("_IA")) 
                m_predictions = readtable(currentPath, opts_without_index);
                m_predictions = table2array(m_predictions);
            else
                m_predictions = readtable(currentPath, opts_with_index);
                m_predictions = table2array(m_predictions);
            end
    
            % metrikler hesaplanıyor
            % biri matris biri de vektır, acaba ikisi de mi vektor
            % olmalıydı
            [m_Avg, m_results] = Metrics(raw_dataset, m_predictions);
    
            activeFilename_with_no_extension = string(extractBetween(activeFilename,1,strlength(activeFilename) - 4));
            m_param_list = split(activeFilename_with_no_extension,"_");
            m_param_dataset = active_collection;

            m_param_approach = string(extractBetween(m_param_list(2), 1, strlength(m_param_list(2))));
            m_param_alpha = string(extractBetween(m_param_list(3), 1, strlength(m_param_list(3))));
            m_param_randomize_run = string(m_param_list(4));
            m_param_cmethod = string(m_param_list(5)); %calculation method
            m_param_algorithm = string(m_param_list(6));

            m_param_prediction_alg = string(m_param_list(8));
    
            % parametre ve metrikler kaydedilmek üzere map'leniyor.
            m_export_cell{m_file_counter,1} = datetime("now");
            m_export_cell{m_file_counter,2} = m_file_counter;
            m_export_cell{m_file_counter,3} = activeFilename_with_no_extension;
            m_export_cell{m_file_counter,4} = m_param_dataset;% mlm
            m_export_cell{m_file_counter,5} = m_param_approach;% aad approach
            m_export_cell{m_file_counter,6} = m_param_alpha; %010 alpha
            m_export_cell{m_file_counter,7} = m_param_randomize_run;%r1 randomize run
            m_export_cell{m_file_counter,8} = m_param_cmethod; %amain fake rating calculation method
            m_export_cell{m_file_counter,9} = m_param_algorithm;%vaecf
            m_export_cell{m_file_counter,10} = m_param_prediction_alg;% hpf vs vs
            
            m_export_cell{m_file_counter,11} = m_Avg(1);
            m_export_cell{m_file_counter,12} = m_Avg(2);
            m_export_cell{m_file_counter,13} = m_Avg(3);
            m_export_cell{m_file_counter,14} = m_Avg(4);
            m_export_cell{m_file_counter,15} = m_Avg(5);
            m_export_cell{m_file_counter,16} = m_Avg(6);
            m_export_cell{m_file_counter,17} = m_Avg(7);
            m_export_cell{m_file_counter,18} = m_Avg(8);
            m_export_cell{m_file_counter,19} = m_Avg(9);
            m_export_cell{m_file_counter,20} = m_Avg(10);
            m_export_cell{m_file_counter,21} = m_Avg(11);
            m_export_cell{m_file_counter,22} = m_Avg(12);
            
            m_result_output = strcat(datestr(datetime("now"))) + ";" + m_file_counter + ";" + activeFilename_with_no_extension + ";" + m_param_dataset + ";" + m_param_approach + ";" + m_param_alpha + ";" + m_param_randomize_run + ";" + m_param_cmethod + ";" + m_param_algorithm + ";" + m_param_prediction_alg + ";" + m_Avg(1) + ";" + m_Avg(2) + ";" + m_Avg(3) + ";" + m_Avg(4) + ";" + m_Avg(5) + ";" + m_Avg(6) + ";" + m_Avg(7) + ";" + m_Avg(8) + ";" + m_Avg(9) + ";" + m_Avg(10) + ";" + m_Avg(11) + ";" + m_Avg(12);
            m_saveFilePath = strcat("../out/9_evaluations/", active_collection, "/", activeFilename_with_no_extension, ".mat");
            save(m_saveFilePath, 'm_results', '-v7.3');
    
            disp(m_result_output);

            % if m_file_counter == 5
            %     disp("done")
            % end

        end
    end

    m_table_header = {'date', 'counter', 'filename', 'dataset', 'approach', 'alpha', 'randomize', 'calculation', 'algorithm', 'prediction', 'GAPp', 'GAPr', 'DeltaGAP_IM', 'DeltaGAP_MA', 'NCDG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelity', 'LTC', 'Entropy'};
    m_results_table = cell2table(m_export_cell,'VariableNames', m_table_header);
    m_results_table_path = strcat("../out/9_evaluations/", active_collection, "_all_", datestr(datetime("now"),'yyyy-dd-mm'),".csv");
    writetable(m_results_table, m_results_table_path,'Delimiter',';');

end

disp("Finish e_9_evaluations")