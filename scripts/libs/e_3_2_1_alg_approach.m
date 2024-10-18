% 2023_11_02 vesiyonu tamamlandı

% burada calc_rating altındaki MLM_algorithm_vaecf dosyası üretilmesi 
% oncesinde veri csv uzantılı vektor olarak kaydedilecek

% sonrasında amac, tüm ratingleri algoritmaya urettirip, 
% -1 olarak fake rating
% bekleyen alanlara algoritma tahminlerini ekletmek
% buradan sonra artık pythonda prediction yapılacak


clc;
clear;

disp("Start e_3_2_1_alg_approach");

m_dataset_name = "MLM";
m_dataset_name = "Yelp";
m_dataset_name = "DoubanBooks";


m_dest = strcat('../out/3_2_0_algorithm_approach_raw/',m_dataset_name,"/");

% ham veriyi workplace'e alıyorum
copyfile('../out/0_0_prepare_input/*.mat', m_dest);

% oncelikle ham veri alındı ve csv olarak kaydedildi
% format, vektör

m_path = strcat('../out/3_2_0_algorithm_approach_raw/',m_dataset_name,"/",m_dataset_name,'.mat');
m_save_path = strcat('../out/3_2_1_algorithm_approach_convert_csv/',m_dataset_name, '/',m_dataset_name','.csv');

% veriyi oku
m_DataSet = load(m_path);
temp_DataSet = struct2cell(m_DataSet);
m_DataSet = temp_DataSet{1};

writematrix(m_DataSet,m_save_path);

disp("Finish e_3_2_1_alg_approach");

