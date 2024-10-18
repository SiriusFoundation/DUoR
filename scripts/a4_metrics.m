clc
clear

disp("evaluation başladı");

% m_dataset = 'MLM';
% m_dataset = 'Yelp';
m_dataset = 'doubanbook';

m_dataset_raw_name = "DoubanBooks";

m_raw_dataset = load(strcat('../output/0_datasets/',m_dataset_raw_name, ".mat"));
temp_raw_dataset = struct2cell(m_raw_dataset);
m_raw_dataset = temp_raw_dataset {1};

m_ImportPath = "../output/3_debiasing/";
m_ExportPath = "../output/4_evaluations/";

m_reach_file_path = strcat(m_ImportPath, m_dataset, "\*.mat");
m_fileList = dir(m_reach_file_path);


m_export_cell = cell(size(m_fileList,1),15);
m_folder_name = [m_fileList(1).folder];
for m_file_counter = 1:length(m_fileList)
    m_file_name = [m_fileList(m_file_counter).name];
    m_active_path = strcat(m_folder_name, "\", m_file_name);

    m_top_n = load(m_active_path);
    temp_top_n = struct2cell(m_top_n);
    m_top_n = temp_top_n {1};

    [m_Avg, m_results] = Metrics(m_raw_dataset, m_top_n);

    %(1-BTA, 2-APRI, 3-RMSE-PC, 4-NDCG, 5-Precision, 6-Recall, 7-F1, 8-APLT, 9-Novelty, 10-MRM 11-LTC, 12-Entropy)

    m_export_cell{m_file_counter,1} = datetime("now");
    m_export_cell{m_file_counter,2} = m_file_counter;
    m_export_cell{m_file_counter,3} = m_file_name;

    m_export_cell{m_file_counter,4} = m_Avg(1); % BTA
    m_export_cell{m_file_counter,5} = m_Avg(2); % APRI
    m_export_cell{m_file_counter,6} = m_Avg(3); % RMSE
    m_export_cell{m_file_counter,7} = m_Avg(4); % NDCG
    m_export_cell{m_file_counter,8} = m_Avg(5); % Precision
    m_export_cell{m_file_counter,9} = m_Avg(6); % Recall
    m_export_cell{m_file_counter,10} = m_Avg(7); % F1
    m_export_cell{m_file_counter,11} = m_Avg(8); % APLT
    m_export_cell{m_file_counter,12} = m_Avg(9); % Novelty
    m_export_cell{m_file_counter,13} = m_Avg(10); % MRM
    m_export_cell{m_file_counter,14} = m_Avg(11); % LTC
    m_export_cell{m_file_counter,15} = m_Avg(12); % Entropy

    m_result_output = strcat(num2str(m_file_counter), " - ", m_file_name, " - done");
    m_file_name_with_no_ext = m_file_name(1:end-4);
    m_saveFilePath = strcat(m_ExportPath, m_file_name_with_no_ext, "_results.mat");
    save(m_saveFilePath, 'm_results', '-v7.3');

    disp(m_result_output);

end

m_table_header = {'date', 'counter', 'filename', 'BTA', 'APRI', 'RMSE-PC', 'NDCG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelty', 'MRM', 'LTC', 'Entropy'};
m_results_table = cell2table(m_export_cell,'VariableNames', m_table_header);
m_results_table_path = strcat(m_ExportPath, m_dataset, "_all_", datestr(datetime("now"),'yyyy-dd-mm'),".csv");
writetable(m_results_table, m_results_table_path,'Delimiter',';');

disp("evaluation bitti");
