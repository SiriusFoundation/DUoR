% 2023_11_02 vesiyonu tamamlandı

% bu adımda fake ratingler üretiliyor,
% aslında veri setinin tamamı fake rating, 
% apply rating adımında bunlar veri setlerine inject edilecekler
% veri setleri kontorl edilirse systemde degeler sistem oratalması, itemda
% kolon değerleri item oratalaması, user da satırlar userların ortlaaması

% projedeki user, item ve system rating ortalaması olan degerler ekleniyor
% toplamda 3 farklı dosya üretilecek, user, sistem, item
% dizinde yer alan 4. dosya olan MLM_algorithm_vaecf dosyası
% farklı script tarafından üretiliyor
% emre hoca ile burası da kontrol edildi ve ok, problem yok

clc;
clear;

disp("Start e_3_1_1_basic_approach");

m_dataset = 'MLM';
m_dataset = 'Yelp';
m_dataset = 'DoubanBooks';


m_path = strcat('../out/1_2_prepare_uir/',m_dataset,"/",m_dataset,'.mat');

% gerçek, ham veri seti okundu
DataSet_UIR = load(m_path);
temp_DataSet = struct2cell(DataSet_UIR);
DataSet_UIR = temp_DataSet{1};
clear temp_DataSet;

% sistem, user ve item olacak şekilde 3 farklı kopya alındı
DataSet_Average_System = DataSet_UIR;
Dataset_Average_User = DataSet_UIR;
Dataset_Average_Item = DataSet_UIR;

% döngüler için user ve item sayıları alındı
m_user_count = size(DataSet_UIR,1);
m_item_count = size(DataSet_UIR,2);

% system rating hesaplanıyor
m_system_rating_count = 0;
m_system_rating_sum = 0;

% sistemin rating ortalaması hesaplanıyor
% önce toplamı, sonra da rating sayısı, sonra da ortalama alınacak
for i = 1:m_user_count
    for j = 1:m_item_count
        m_active_rating = DataSet_UIR(i,j);
        if (m_active_rating ~= 0)
            m_system_rating_count = m_system_rating_count + 1;
            m_system_rating_sum = m_system_rating_sum + m_active_rating;
        end
    end
end

% sistemin ortalama rating degeri
m_system_rating = m_system_rating_sum / m_system_rating_count;

% sistem rating değeri içeren matris olusturuluyor
for i = 1:m_user_count
    for j = 1:m_item_count
        DataSet_Average_System(i,j) = m_system_rating;
    end
end

% 0/0'dan gelen ortalamada NaN degerleri 0 a cevrildi
DataSet_Average_System = fillmissing(DataSet_Average_System, 'constant', 0);

m_method_string = "system_rating_mean";

m_filename = strcat(m_dataset, "_", m_method_string, ".mat");
m_filepath = strcat("../out/3_1_1_basic_approach/",m_dataset,"/",m_filename);
disp(m_filepath);
save(m_filepath, 'DataSet_Average_System', '-v7.3');


% user rating hesaplanıyor
for i = 1:m_user_count

    m_user_rating_count = 0;
    m_user_rating_sum = 0;

    for j = 1:m_item_count
        m_active_rating = DataSet_UIR(i,j);
        if (m_active_rating ~= 0)

            m_user_rating_count = m_user_rating_count + 1;
            m_user_rating_sum = m_user_rating_sum + m_active_rating;

        end
    end

    m_user_rating = m_user_rating_sum / m_user_rating_count;

    for j = 1:m_item_count
        Dataset_Average_User(i,j) = m_user_rating;
    end
    
end

% 0/0'dan gelen ortalamada NaN'lar 0 a cevrildi
Dataset_Average_User = fillmissing(Dataset_Average_User, 'constant', 0);

% teknik olarak satırların aynı oldugu bir matris user average matrisi

m_method_string = "user_rating_mean";

m_filename = strcat(m_dataset, "_", m_method_string, ".mat");
m_filepath = strcat("../out/3_1_1_basic_approach/",m_dataset,"/",m_filename);
disp(m_filepath);
save(m_filepath, 'Dataset_Average_User', '-v7.3');

% user rating hesaplandı

% item rating hesaplanıyor
% buradaki kodu anlamazsan eger, 
% itemlar için aslında döngüyü tersten döndürmedim,
% matrisin transpozesini alarak iuserdaki işlemleri aynen yaptım

DataSet_UIR = transpose(DataSet_UIR);
Dataset_Average_Item = transpose(Dataset_Average_Item);

% user item degerlerini switch yaptım
temp_count = m_user_count;
m_user_count = m_item_count;
m_item_count = temp_count;

for i = 1:m_user_count

    m_item_rating_count = 0;
    m_item_rating_sum = 0;

    for j = 1:m_item_count
        m_active_rating = DataSet_UIR(i,j);
        if (m_active_rating ~= 0)

            m_item_rating_count = m_item_rating_count + 1;
            m_item_rating_sum = m_item_rating_sum + m_active_rating;

        end
    end

    m_item_rating = m_item_rating_sum / m_item_rating_count;

    for j = 1:m_item_count
        Dataset_Average_Item(i,j) = m_item_rating;
    end
    
end

% 0/0'dan gelen ortalamada NaN'lar 0 a cevrildi
Dataset_Average_Item = fillmissing(Dataset_Average_Item, 'constant', 0);

DataSet_UIR = transpose(DataSet_UIR);
Dataset_Average_Item = transpose(Dataset_Average_Item);

temp_count = m_user_count;
m_user_count = m_item_count;
m_item_count = temp_count;

m_method_string = "item_rating_mean";

m_filename = strcat(m_dataset, "_", m_method_string, ".mat");
m_filepath = strcat("../out/3_1_1_basic_approach/",m_dataset,"/",m_filename);
disp(m_filepath);
save(m_filepath, 'Dataset_Average_Item', '-v7.3');

% item rating hesaplandı

disp("Finish e_3_1_1_basic_approach");