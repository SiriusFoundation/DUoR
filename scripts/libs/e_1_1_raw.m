% 2023_11_02 vesiyonu tamamlandı
% create_1_raw.m
% verinin işlem başlaması öncesinde ilk kopyalanması işlemi

clc;
clear;

disp("Start e_1_1_raw");

% ham veriyi workplace'e alıyorum
% yedek amacıyla kopyalama işlemi

m_dataset_name = "MLM";
m_dataset_name = "Yelp";
m_dataset_name = "DoubanBooks";

m_path = strcat('../out/1_1_prepare_raw/',m_dataset_name);

copyfile('../out/0_0_prepare_input/*.mat', m_path);

disp("Finish e_1_1_raw");