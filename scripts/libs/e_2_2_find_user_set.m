% 2023_11_02 vesiyonu tamamlandı
% e4_find_user_set.m
% 19 satırlık veri setine 24 satır daha ekleniyor, bunun nedeni 8 alpha
% için 3 er yaklaşımdan 24 farklı fake rating eklenecek userlar
% belirlenecek
% burada hangi userlar degil, kaç user'a ekleneceği hesaplanıyor.
% bir sonraki adımda hangi userlar oldugu belirlenecek

% okunan veri seti 
% veri seti
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


clc;
clear;

disp("Start e_2_2_find_user_set");

m_dataset = 'MLM';
m_dataset = 'Yelp';
m_dataset = 'DoubanBooks';


m_path = strcat('../out/2_1_find_unpopular/',m_dataset,'/unpopular_items.mat');

DataSet_Summary = load(m_path);
temp_DataSet = struct2cell(DataSet_Summary);
DataSet_Summary = temp_DataSet{1};

clear temp_DataSet;

% 18 m_count_item
m_count_item_total = DataSet_Summary(18,1);

% 19 m_count_user
m_count_user = DataSet_Summary(19,1);

% 4 Total Rating Count
m_count_total_rating = DataSet_Summary(4,1);

% 15 AAD_Item_Count
m_count_m_AAD_item = DataSet_Summary(15,1);
% 16 HAD_Item_Count
m_count_m_HAD_item = DataSet_Summary(16,1);
% 17 PAD_Item_Count
m_count_m_PAD_item = DataSet_Summary(17,1);

% 13 PAD Threshold (E)
m_E_PAD_Threshold = DataSet_Summary(13,1);

% 11 Head Items Average Rating Count (POPb)
m_POPB_Head_Item_Average_Rating_Count = DataSet_Summary(11,1);

% 5 Average Rating Count (POPg)
m_POPG_Average_Rating_Count = DataSet_Summary(5,1);

% 9 Pareto tail item count
m_count_item_head = DataSet_Summary(9,1);
m_count_item_tail = m_count_item_total - m_count_item_head;

% 8 tane alpha parametresi var
m_alpha = [0.1, 0.25, (1 / 3), 0.5, (2 / 3), 0.75, 0.9, 1];

% aad, had, pad diye de 3 yöntem var
m_DataSet_Alphas = zeros(24,m_count_item_total,'int32');

% bu döngüdeki amaç, ilgili itemda kaç kullanıcıya fake rating eklenmesi
% gerektiği hesaplanıyor, tabi alpha katsayısına göre her item için 8
% farklı değer hesaplanacak
for i=1:m_count_item_total

    % aad
    sil_aad = DataSet_Summary(10,i);
    if (DataSet_Summary(10,i) == 1)
                      % 5 Average Rating Count (POPg) - % 3 Rating Count (Popü)
        m_DataSet_Alphas(1,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(1);
        m_DataSet_Alphas(2,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(2);
        m_DataSet_Alphas(3,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(3);
        m_DataSet_Alphas(4,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(4);
        m_DataSet_Alphas(5,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(5);
        m_DataSet_Alphas(6,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(6);
        m_DataSet_Alphas(7,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(7);
        m_DataSet_Alphas(8,i) = (DataSet_Summary(5,i) - DataSet_Summary(3,i)) * m_alpha(8);
    end

    % had
    sil_had = DataSet_Summary(12,i);
    if (DataSet_Summary(12,i) == 1)
           % 11 Head Items Average Rating Count (POPb) - % 3 Rating Count (Popü)
        m_DataSet_Alphas(9,i) =  (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(1);
        m_DataSet_Alphas(10,i) = (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(2);
        m_DataSet_Alphas(11,i) = (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(3);
        m_DataSet_Alphas(12,i) = (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(4);
        m_DataSet_Alphas(13,i) = (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(5);
        m_DataSet_Alphas(14,i) = (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(6);
        m_DataSet_Alphas(15,i) = (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(7);
        m_DataSet_Alphas(16,i) = (DataSet_Summary(11,i) - DataSet_Summary(3,i)) * m_alpha(8);
    end

    % pad
    sil_pad = DataSet_Summary(14,i);
    if (DataSet_Summary(14,i) == 1)
                                 % 13 PAD Threshold (E) - % 3 Rating Count (Popü)
        m_DataSet_Alphas(17,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(1);
        m_DataSet_Alphas(18,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(2);
        m_DataSet_Alphas(19,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(3);
        m_DataSet_Alphas(20,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(4);
        m_DataSet_Alphas(21,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(5);
        m_DataSet_Alphas(22,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(6);
        m_DataSet_Alphas(23,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(7);
        m_DataSet_Alphas(24,i) = (DataSet_Summary(13,i) - DataSet_Summary(3,i)) * m_alpha(8);
    end
end

% 24lük ve 19'luk veri seti birleştiriliyor
Dataset_Combined = vertcat(DataSet_Summary,m_DataSet_Alphas);

m_save_path = strcat('../out/2_2_find_user/', m_dataset,'/unpopular_items_with_alphas.mat');
save(m_save_path, 'Dataset_Combined', '-v7.3');

disp("Finish e_2_2_find_user_set");
