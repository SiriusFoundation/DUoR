clc;
clear;

disp("Start all debiasing");

m_filename_raw_dataset = "MLM.mat";
m_filename_raw_dataset = "Yelp.mat";
m_filename_raw_dataset = "DoubanBooks.mat";

m_dataset_predictions = "MLM_SKM";
m_dataset_predictions = "MLM_HPF";
m_dataset_predictions = "YL_SKM";
m_dataset_predictions = "YL_HPF";
m_dataset_predictions = "DB_SKM";
m_dataset_predictions = "DB_HPF";

m_filename_predictions = strcat(m_dataset_predictions,".mat");

m_path_raw_dataset = strcat('../output/0_datasets/', m_filename_raw_dataset);
m_path_predictions = strcat('../output/2_predictions/', m_filename_predictions);

m_raw_dataset = load(m_path_raw_dataset);
temp_raw_dataset = struct2cell(m_raw_dataset);
m_raw_dataset = temp_raw_dataset{1};

m_predictions = load(m_path_predictions);
temp_predictions = struct2cell(m_predictions);
m_predictions = temp_predictions{1};

% ------------------ debiasing başlıyor ----------------------------

% BTA
m_path_save = strcat('../output/3_debiasing/',"BetterThanAverage_", m_dataset_predictions);
m_save_path_csv = strcat(m_path_save, ".csv");
m_save_path_mat = strcat(m_path_save, ".mat");

[BetterThanAverage_TopNRecs] = debiasing_BTA(m_raw_dataset, m_predictions, 100, 10);
writematrix(BetterThanAverage_TopNRecs, m_save_path_csv);
save(m_save_path_mat, 'BetterThanAverage_TopNRecs');
disp("BTA_TopNRecs Export Edildi");

% CP
m_path_save = strcat('../output/3_debiasing/',"CalibratedPopularity_", m_dataset_predictions);
m_save_path_csv = strcat(m_path_save, ".csv");
m_save_path_mat = strcat(m_path_save, ".mat");

[CalibratedPopularity_TopNRecs] = debiasing_CP(m_raw_dataset, m_predictions, 100, 10);
writematrix(CalibratedPopularity_TopNRecs, m_save_path_csv);
save(m_save_path_mat, 'CalibratedPopularity_TopNRecs');
disp("CP_TopNRecs Export Edildi");

%xQuad
m_path_save = strcat('../output/3_debiasing/',"xQuad_", m_dataset_predictions);
m_save_path_csv = strcat(m_path_save, ".csv");
m_save_path_mat = strcat(m_path_save, ".mat");

[xQuad_TopNRecs] = debiasing_xQuad(m_raw_dataset, m_predictions, 100, 10);
writematrix(xQuad_TopNRecs, m_save_path_csv);
save(m_save_path_mat, 'xQuad_TopNRecs');
disp("xQuad_TopNRecs Export Edildi");

%Var Mul ERP
% [VarTopN, MulTopN, AugTopN]
m_path_save_var = strcat('../output/3_debiasing/',"VAR_", m_dataset_predictions);
m_save_path_csv_var = strcat(m_path_save_var, ".csv");
m_save_path_mat_var = strcat(m_path_save_var, ".mat");

m_path_save_mul = strcat('../output/3_debiasing/',"MUL_", m_dataset_predictions);
m_save_path_csv_mul = strcat(m_path_save_mul, ".csv");
m_save_path_mat_mul = strcat(m_path_save_mul, ".mat");

m_path_save_aug = strcat('../output/3_debiasing/',"AUG_", m_dataset_predictions);
m_save_path_csv_aug = strcat(m_path_save_aug, ".csv");
m_save_path_mat_aug = strcat(m_path_save_aug, ".mat");

[VAR_TopNRecs, MUL_TopNRecs, AUG_TopNRecs] = debiasing_VaR_ERPs(m_raw_dataset, m_predictions, 10);

writematrix(VAR_TopNRecs, m_save_path_csv_var);
save(m_save_path_mat_var, 'VAR_TopNRecs');
disp("VAR_TopNRecs Export Edildi");

writematrix(MUL_TopNRecs, m_save_path_csv_mul);
save(m_save_path_mat_mul, 'MUL_TopNRecs');
disp("MUL_TopNRecs Export Edildi");

writematrix(AUG_TopNRecs, m_save_path_csv_aug);
save(m_save_path_mat_aug, 'AUG_TopNRecs');
disp("AUG_TopNRecs Export Edildi");

% %LNSF
% m_path_save = strcat('../output/3_debiasing/',"LNSF_", m_dataset_predictions);
% m_save_path_csv = strcat(m_path_save, ".csv");
% m_save_path_mat = strcat(m_path_save, ".mat");
% 
% [LNSF_TopNRecs] = debiasing_LNSF(m_raw_dataset, m_predictions, 10);
% writematrix(LNSF_TopNRecs, m_save_path_csv);
% save(m_save_path_mat, 'LNSF_TopNRecs');
% disp("LNSF_TopNRecs Export Edildi");

disp("finish all debiasing");

