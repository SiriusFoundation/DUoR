% 2023_11_02 vesiyonu tamamlandı
% pythonda algoirtma ratingleri predict etti
% burada da predict edilen ratingler hesaplamaya sokulması öncesinde
% normalize edilecek

clc;
clear;

disp("Start e_3_2_3_alg_normalize_rating");

m_dataset_name = "MLM";
m_dataset_name = "Yelp";
m_dataset_name = "DoubanBooks";


m_algorithm_name = 'vaecf';

% path belirlendi
m_path = strcat('../out/3_2_2_algorithm_approach_predict_with_algoritm/',m_dataset_name,'/',m_dataset_name,'_',m_algorithm_name,'.csv');
m_save_path = strcat('../out/3_2_3_algorithm_approach_normalize/',m_dataset_name,'/',m_dataset_name,'_',m_algorithm_name,'.mat');

% veri okunuyor
disp("Start reading");
m_Data_Set = readmatrix(m_path);

% veri okundu ancak formatı temizlemek gerekiyor
% sol solonda index var, ilk satırda da kolon başlıkları var
m_Data_Set(1,:) = [];
m_Data_Set(:,1) = [];
disp("Finish reading");
% artık normal vektör formatına dönüldü

% normalizasyona hazırlık
% normalizasyona hazırlık için min ve max degerler alınıyor
m_ratings = m_Data_Set(:,3);
m_global_min = min(m_ratings);
m_global_max = max(m_ratings);

m_row_size = size(m_Data_Set,1);
m_column_size = size(m_Data_Set,2);

m_new_Data_Set = m_Data_Set;

% veri normalize ediliyor
disp("Start normalizing");
for i=1:m_row_size
    m_actual_prediction = m_Data_Set(i,3);

    % 0 ile 1 arasına normalize edildi
    m_normalized_rating = ((m_actual_prediction - m_global_min) / (m_global_max - m_global_min));

    % rating skalası 1 ile 5, bu nedenle 5 ile çarpıldı
    m_normalized_rating = m_normalized_rating * 5;

    % küsürat atıldı
    m_normalized_rating = round(m_normalized_rating,2);

    % son deger atandı
    m_new_Data_Set(i,3) = m_normalized_rating;
end
disp("Finish normalizing");

% sağlama ve kontrol
m_min = min(m_new_Data_Set(:,3));
m_max = max(m_new_Data_Set(:,3));
m_new_ratings_ordered = m_new_Data_Set(:,3);
m_new_ratings_ordered = sort(m_new_ratings_ordered,'descend');
% sağlama ve kontrol bitti

disp("Start normalize exporting");
save(m_save_path, 'm_new_Data_Set', '-v7.3');
disp("Finish normalize exporting");

% veri vektorden matrise cevriliyor
m_path = m_save_path;
m_save_path = strcat('../out/3_2_4_algorithm_approach_uir/',m_dataset_name,'/',m_dataset_name,'_algorithm_',m_algorithm_name,'.mat');

disp("Start read vector");
% az once kaydedilen vektör formatındaki veriyi oku
m_DataSet = load(m_path);
disp("finish read vector");

disp("Start matrix generation");
% generate_uir fonksiyonu ile matrisi generate et
m_uir = generate_uir(m_DataSet);
disp("Finish matrix generation");

disp("Start export matrix");
% matrisi kaydet ( veri büyük olunca v7.3 üstü ile kaydetmek gerekti
save(m_save_path, 'm_uir', '-v7.3');
disp("Finish export matrix");

disp("Finish e_3_2_3_alg_normalize_rating");