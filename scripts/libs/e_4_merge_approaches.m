% 2023_11_02 vesiyonu tamamlandı
% burada tüm user item matrislerine 
% fake ratingler inject edilmesi öncesinde  
% aynı dizin altına toplanıyor

clc;
clear;

disp("e_4_merge_approaches started!");

m_dataset_name = "MLM";
m_dataset_name = "Yelp";
m_dataset_name = "DoubanBooks";


m_source_basic = strcat("../out/3_1_1_basic_approach","/",m_dataset_name,"/*");
m_source_algorithm = strcat("../out/3_2_4_algorithm_approach_uir","/",m_dataset_name,"/*");
m_source_disguise1 = strcat("../out/3_3_2_disguise_denormalize","/",m_dataset_name,"/DoubanBooks_disg_S2B5N.mat");
m_source_disguise2 = strcat("../out/3_3_2_disguise_denormalize","/",m_dataset_name,"/DoubanBooks_disg_S3B10N.mat");
m_source_disguise3 = strcat("../out/3_3_2_disguise_denormalize","/",m_dataset_name,"/DoubanBooks_disg_S4B25N.mat");

m_destination = strcat("../out/4_merge_approaches","/",m_dataset_name,"/");

% basic
copyfile(m_source_basic, m_destination);

% algorithm
copyfile(m_source_algorithm, m_destination);

% disguise
copyfile(m_source_disguise1, m_destination);
copyfile(m_source_disguise2, m_destination);
copyfile(m_source_disguise3, m_destination);

disp("e_4_merge_approaches finished!");