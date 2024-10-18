clc;
clear;

disp("Start xQuad");

m_filename_raw_dataset = "MLM.mat";
m_filename_raw_dataset = "Yelp.mat";
m_filename_raw_dataset = "DoubanBooks.mat";

m_filename_predictions = "MLM_SKM.mat";
m_filename_predictions = "MLM_HPF.mat";
m_filename_predictions = "YL_SKM.mat";
m_filename_predictions = "YL_HPF.mat";
m_filename_predictions = "DB_SKM.mat";
m_filename_predictions = "DB_HPF.mat";

m_save_path = "DB_HPF";


m_path_raw_dataset = strcat('../output/0_datasets/', m_filename_raw_dataset);
m_path_predictions = strcat('../output/2_predictions/', m_filename_predictions);
m_path_save = strcat('../output/3_debiasing/',"xQuad", m_save_path);
m_save_path_csv = strcat(m_path_save, ".csv");
m_save_path_mat = strcat(m_path_save, ".mat");

m_raw_dataset = load(m_path_raw_dataset);
temp_raw_dataset = struct2cell(m_raw_dataset);
m_raw_dataset = temp_raw_dataset{1};

m_predictions = load(m_path_predictions);
temp_predictions = struct2cell(m_predictions);
m_predictions = temp_predictions{1};

[CalibratedPopularity_TopNRecs] = debiasing_CalibratedPopularity(m_raw_dataset, m_predictions, 100, 10);

% CSV dosyasına kaydetme
writematrix(CalibratedPopularity_TopNRecs, m_save_path_csv);

% MATLAB dosyasına kaydetme
save(m_save_path_mat, 'CalibratedPopularity_TopNRecs');

disp("Finish xQuad");

