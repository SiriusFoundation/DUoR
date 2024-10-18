% 2023_11_02 vesiyonu tamamlandı
% create_2_uir.m
% bu aşamada 3 farklı yaklaşım ile populer ve unpopuler itemları
% ayrıştırıyorum aslında çalışmadaki 2 ana başlıktan biri tamamen burada

clc;
clear;

disp("Start e_2_1_find_unpopular");

% veri okundu
m_dataset = 'MLM';
m_dataset = 'Yelp';
m_dataset = 'DoubanBooks';


m_path = '../out/1_2_prepare_uir/';
m_path = strcat(m_path,m_dataset,"/",m_dataset,".mat");

DataSet_UIR = load(m_path);
temp_DataSet = struct2cell(DataSet_UIR);
DataSet_UIR = temp_DataSet{1};

% user item sayıları alındı
m_count_user = size(DataSet_UIR,1);
m_count_item = size(DataSet_UIR,2);

% veri seti oluşturuldu
m_pop = zeros(19,m_count_item,'int32');

% kaydedilecek veri seti formatı
% 1 Data Order
% 2 Pareto Order
% 3 Rating Count (Popü)
% 4 Total Rating Count
% 5 Average Rating Count (POPg)
% 6 Pareto Head Volume
% 7 Head or Tail Cumulative Sum
% 8 Head or Tail
% 9 Pareto tail item count
% 10 AAD Candidate
% 11 Head Items Average Rating Count (POPb)
% 12 HAD Candidate
% 13 PAD Threshold (E)
% 14 PAD Candidate
% 15 AAD_Item_Count
% 16 HAD_Item_Count
% 17 PAD_Item_Count
% 18 m_count_item
% 19 m_count_user

% 3 yontem var
% AAD - All Average Difference
% HAD - Head Average Difference
% PAD - Pareto Average Difference


% AAD - All Average Difference

% toplam rating sayısı
m_all_item_total_rating_count = 0;

% verilen rating sayıları hesaplanıyor
for i=1:m_count_item
    for j=1:m_count_user
        m_rating = DataSet_UIR(j,i);
        if (m_rating ~= 0)

            % itemın aldığı toplam rating sayıları hesaplanıyor
            m_pop(3,i) = m_pop(3,i) + 1;

        end
    end
    % ortalama rating sayısı için genel toplam hesaplanıyor
    m_all_item_total_rating_count = m_all_item_total_rating_count + m_pop(3,i);
end

% sistemdeki ortalama rating sayısı
m_popg = m_all_item_total_rating_count / m_count_user;

% paretoya gore toplam rating sayısının %20'si
% yani head kısmının toplam rating hacmi
m_pareto_head_volume = m_all_item_total_rating_count * 0.2;

% Data Order işlemi, 
% gerçek veri sort edildiğine bozulmasın diye 
% index ekleniyor
for i=1:m_count_item
    % 1 e sıralama verildi
    m_pop(1,i) = i;

    % toplam rating sayısı atanıyor
    m_pop(4,i) = m_all_item_total_rating_count;

    % popg yani sistemdeki ortalama rating sayısı atanıyor
    m_pop(5,i) = m_popg;

    % head itemların popüler olması için 
    % gereken rating kümesi büyüklüğü yazılıyor
    m_pop(6,i) = m_pareto_head_volume;
end

% burada pareto ordera göre tekrar sıralandı,
m_popT = transpose(m_pop);
m_popT = sortrows(m_popT,3,'descend');
m_pop = transpose(m_popT);

%burada head ve tail itemlarla ilgili hesaplamalar yapılacak
% head kısmındaki toplam rating sayısı
m_total_rating_count_for_head = 0;

% head kısmındaki toplam item sayısı
m_tail_item_count = 0;

for i=1:m_count_item

    % pareto order için index eklendi
    m_pop(2,i) = i;

    % en baştan itibaren ratingler toplanıyor
    m_total_rating_count_for_head = m_total_rating_count_for_head + m_pop(3,i);

    % toplam rating değeri sistemdeki toplam rating değerinin %20'sine
    % gelene kadar itemlar head, 
    % sonrası için de itemlar tail diye işaretleniyor
    if (m_total_rating_count_for_head <= m_pareto_head_volume)
        % head
        % 8 Head or Tail
        m_pop(8,i) = 0;
    else
        % tail
        % 8 Head or Tail
        m_pop(8,i) = 1;

        % ayrıca tail item sayısını count ettik
        m_tail_item_count = m_tail_item_count + 1;
    end

    % doğru hesaplamak için kümülatif rating sayısını yazdırdım
    % 7 Head or Tail Cumulative Sum
    m_pop(7,i) = m_total_rating_count_for_head;

end

% head tail sınırındaki itemı arıyoruz
m_head_threshold_item = 0;

for i=1:m_count_item
    % tail item sayısını yazdırıyoruz
    % 9 Pareto tail item count
    m_pop(9,i) = m_tail_item_count;

    % find head threshold item
    if (m_pop(8,i) == 0)
        m_head_threshold_item = i;    
    end
end

% item aad adayı mıdır flagleniyor
for i=1:m_count_item
    % AAD Candidate
    if (m_pop(5,i) - m_pop(3,i) > 0)
        % 8 Head or Tail
        m_pop(10,i) = 1;
    end
end

% head itemların aldığı rating sayısı hesaplanacak
m_head_item_total_rating_count = 0;
m_head_item_count = m_count_item - m_tail_item_count;

for i=1:m_count_item

    % item head ise
    if (m_pop(8,i) == 0)
        m_head_item_total_rating_count = m_head_item_total_rating_count + m_pop(3,i);
    end

end

% headlerin ortalaması
m_average_head_item_rating_count = m_head_item_total_rating_count / m_head_item_count;

% emre hoca ile düzeltme yapıldı ve problem yok
% bu kısmı neden yaptık hatırlayamadım
for i=1:m_count_item

    % popb
    % 11 Head Items Average Rating Count (POPb)
    m_pop(11,i) = m_average_head_item_rating_count;

    % had ve pad'de tailin tamamı olması, head e uygulamaması gerekiyor.
    if (i > m_head_threshold_item)

        % E 
        % direk olarak m_head_threshold_item değerini değil, 
        % o indisteki degeri aldık
        % 13 PAD Threshold (E)
        temp_value = m_pop(3,m_head_threshold_item);
        m_pop(13,i) = temp_value;
    
        % HAD Candidate
        if (m_pop(11,i) - m_pop(3,i) > 0)
            % 12 HAD Candidate
            m_pop(12,i) = 1;
        end
    
        % PAD Candidate
        if (m_pop(13,i) - m_pop(3,i) > 0)
            % 14 PAD Candidate
            m_pop(14,i) = 1;
        end
    end
end

AAD_Item_Count = 0;
HAD_Item_Count = 0;
PAD_Item_Count = 0;

for i=1:m_count_item

    if (m_pop(10,i) == 1)
        AAD_Item_Count = AAD_Item_Count + 1;
    end

    if (m_pop(12,i) == 1)
        HAD_Item_Count = HAD_Item_Count + 1;
    end
    
    if (m_pop(14,i) == 1)
        PAD_Item_Count = PAD_Item_Count + 1;
    end
end

% 15 AAD_Item_Count
% 16 HAD_Item_Count
% 17 PAD_Item_Count
% 18 m_count_item
% 19 m_count_user
for i=1:m_count_item
    m_pop(15,i) = AAD_Item_Count;
    m_pop(16,i) = HAD_Item_Count;
    m_pop(17,i) = PAD_Item_Count;
    m_pop(18,i) = m_count_item;
    m_pop(19,i) = m_count_user;
end

% veri setini eski haline getiriyor
%m_popT = transpose(m_pop);
%m_popT = sortrows(m_popT,1);
%m_pop = transpose(m_popT);

m_pop_analyze = m_pop;

% temizlik için, yeni format için
m_pop(1,:) = [];
m_pop(1,:) = [];
m_pop(2,:) = [];
m_pop(3,:) = [];
m_pop(3,:) = [];
m_pop(3,:) = [];
m_pop(3,:) = [];
m_pop(1,:) = [];
m_pop(1,:) = [];
m_pop(2,:) = [];
m_pop(3,:) = [];

m_pop(4,:) = [];
m_pop(4,:) = [];
m_pop(4,:) = [];

m_save_path = strcat("../out/2_1_find_unpopular/",m_dataset,"/unpopular_items.mat");
save(m_save_path, 'm_pop_analyze', '-v7.3');

% buna gerek olmayabilir mi?
% m_save_path2 =  strcat("../out/3_find_unpopular/",m_dataset,"/unpopular.mat");
% save(m_save_path2, 'm_pop', '-v7.3');

disp("m_pop Created!");

disp("Finish e_2_1_find_unpopular");
