% e_8_0_other_debiasing_prepare
% 	mat dosyalarını csv'ye cevir
% 	mat al
% 	csv ver

%     F:\a_phase_II\out\1_1_prepare_raw\MLM adresinden oku

clc;
clear;

disp("Start e_8_0_other_debiasing_prepare");



m_dataset_name = "DoubanBooks";
m_dataset_name = "Yelp";
m_dataset_name = "MLM";

% oncelikle ham veri alındı ve csv olarak kaydedildi
% format, vektör

m_path = strcat('../out/1_1_prepare_raw/',m_dataset_name,"/",m_dataset_name,'.mat');
m_save_path = strcat('../out/8_1_other_debiasing_predictions/',m_dataset_name, '/',m_dataset_name','.csv');

% veriyi oku
m_DataSet = load(m_path);
temp_DataSet = struct2cell(m_DataSet);
m_DataSet = temp_DataSet{1};

writematrix(m_DataSet,m_save_path);

disp("Finish e_8_0_other_debiasing_prepare");