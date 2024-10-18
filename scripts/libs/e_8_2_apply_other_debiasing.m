% e_8_2_apply_other_debiasing
% 	prediction ve ham al
% 	topn vs ver

clc;
clear;

disp("Start e_8_2_apply_other_debiasing");



m_dataset_name = "DoubanBooks";
m_dataset_name = "Yelp";
m_dataset_name = "MLM";


m_raw_path = strcat('../out/1_2_prepare_uir/',m_dataset_name,"/",m_dataset_name,'.mat');
m_predictions_path = strcat('../out/8_2_apply_other_debiasing/',m_dataset_name,"/",m_dataset_name,'_WBPR.csv');

% veriyi oku
m_raw_dataset = load(m_raw_path);
temp_DataSet = struct2cell(m_raw_dataset);
m_raw_dataset = temp_DataSet{1};

m_predictions = readtable(m_predictions_path);
m_predictions = m_predictions(:,[2,3,4]);
m_predictions = m_predictions{:,:};

disp("x_debiasing_xQuad started");
[xQuad_TopNRecs] = x_debiasing_xQuad(m_raw_dataset, m_predictions, 100, 10);

m_debias_file_name = "xQuad";
m_save_path = strcat('../out/8_3_debiasing_evaluations/',m_dataset_name, '/',m_dataset_name',"_", m_debias_file_name,'.csv');
writematrix(xQuad_TopNRecs, m_save_path);




disp("erp started");
[ERP_VarTopN, ERP_MulTopN, ERP_AugTopN] = x_debiasing_VaR_ERPs(m_raw_dataset, m_predictions, 10);

m_debias_file_name = "ERP_Var";
m_save_path = strcat('../out/8_3_debiasing_evaluations/',m_dataset_name, '/',m_dataset_name',"_", m_debias_file_name,'.csv');
writematrix(ERP_VarTopN, m_save_path);

m_debias_file_name = "ERP_Mul";
m_save_path = strcat('../out/8_3_debiasing_evaluations/',m_dataset_name, '/',m_dataset_name',"_", m_debias_file_name,'.csv');
writematrix(ERP_MulTopN, m_save_path);

m_debias_file_name = "ERP_Aug";
m_save_path = strcat('../out/8_3_debiasing_evaluations/',m_dataset_name, '/',m_dataset_name',"_", m_debias_file_name,'.csv');
writematrix(ERP_AugTopN, m_save_path);


% disp("lnsf started");
% [LNSF_TopNRecs] = x_debiasing_LNSF(m_raw_dataset, m_predictions, 10);

% disp("saving started");

% m_debias_file_name = "LNSF";
% m_save_path = strcat('../out/8_3_debiasing_evaluations/',m_dataset_name, '/',m_dataset_name',"_", m_debias_file_name,'.csv');
% writematrix(LNSF_TopNRecs, m_save_path);


disp("Finish e_8_2_apply_other_debiasing");