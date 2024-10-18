% 2023_11_02 vesiyonu tamamlandı
% create_2_uir.m
% vektör formatındaki veri, user item rating matrisine cevrilir

clc;
clear;

disp("Start e_1_2_uir");

% her veri seti uir formatına cevrilecek
m_dataset_name = "MLM";
m_dataset_name = "Yelp";
m_dataset_name = "DoubanBooks";

m_path = strcat('../out/1_1_prepare_raw/',m_dataset_name,"/",m_dataset_name,'.mat');
m_save_path = strcat('../out/1_2_prepare_uir/',m_dataset_name,"/",m_dataset_name,'.mat');


% veriyi oku
m_DataSet = load(m_path);

% generate_uir fonksiyonu ile matrisi generate et
m_uir = generate_uir(m_DataSet);

% matrisi kaydet ( veri büyük olunca v7.3 üstü ile kaydetmek gerekti
save(m_save_path, 'm_uir', '-v7.3');
disp("UIR Created!");

% DataSet_DoubanBooks = load('../out/1_raw/DoubanBooks.mat');
% m_uir = generate_uir(DataSet_DoubanBooks);
% save('../out/2_uir/DoubanBooks.mat', 'm_uir', '-v7.3');
% disp("DoubanBooks UIR Created!");

% DataSet_Yelp = load('../out/1_raw/Yelp.mat');
% m_uir = generate_uir(DataSet_Yelp);
% save('../out/2_uir/Yelp.mat', 'm_uir', '-v7.3');
% disp("Yelp UIR Created!");

% DataSet_Dianping = load('../out/1_raw/Dianping.mat');
% m_uir = generate_uir(DataSet_Dianping);
% save('../out/2_uir/Dianping.mat', 'm_uir', '-v7.3');
% disp("Dianping UIR Created!");

disp("Finish e_1_2_uir");
