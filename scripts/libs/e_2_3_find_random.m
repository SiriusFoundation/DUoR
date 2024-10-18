% 2023_11_02 vesiyonu tamamlandı
% burada ilk kez 3 yaklaşım, 8 parametre için toplam 24 tane farklı veri
% seti uretiliyor, zaten bunların performansları karşılaştırılacak
% bu veri setlerinde itemlara kaçar fake rating eklenecegi bir önceki adımda hesaplanmıştır
% bu adımda he rürün için, daha önceden hesaplanmış fake rating sayısı kadar 
% rasgele user belirlenecek ve bu rating alanlarının ratingleri -1 olarak set edilerek export alınacak
% rating degeri burda eklenmiyor, sadece fake rating eklenecek random
% userlar belirlenecek

clc;
clear;

disp("Start e_2_3_find_random");

m_dataset = 'MLM';
m_dataset = 'Yelp';
m_dataset = 'DoubanBooks';


m_path = strcat('../out/1_2_prepare_uir/',m_dataset,"/",m_dataset,'.mat');

% orijinal veri seti okundu
DataSet_UIR = load(m_path);
temp_DataSet = struct2cell(DataSet_UIR);
DataSet_UIR = temp_DataSet{1};
clear temp_DataSet;

% toplam kullanıcı sayısı alındı
m_user_count = size(DataSet_UIR,1);

% daha önce hesaplanmış veri seti alındı ve okundu
m_path2 = strcat('../out/2_2_find_user/', m_dataset,'/unpopular_items_with_alphas.mat');
DataSet_Master = load(m_path2);
temp_DataSet = struct2cell(DataSet_Master);
DataSet_Master = temp_DataSet{1};
clear temp_DataSet;

% 3 find method var
m_find_method_names = ["aad" "had" "pad"];
c_find_method_count = size(m_find_method_names,2);

% 8 tane alpha parametresi var
m_alpha = [0.1, 0.25, (1 / 3), 0.5, (2 / 3), 0.75, 0.9, 1];
c_alpha_parameter_count = size(m_alpha,2);

% 5 randomize yapılacak
% gerçek deneyde bu 5 yapılacak
m_random_count = 1;

% hesaplanmış veri seti kopyalanıp populariteye gore sıralanıyor
% bu kopya anlık gerekecek
DataSet_Master_Pop_Order = DataSet_Master;
clear DataSet_Master;

% veri setini eski haline getiriyor
DataSet_Master_Data_Order = transpose(DataSet_Master_Pop_Order);
DataSet_Master_Data_Order = sortrows(DataSet_Master_Data_Order,1);
DataSet_Master_Data_Order = transpose(DataSet_Master_Data_Order);

% su anda elimizde hesaplanmış veri seti data order ve popularity order
% olarak 2 farklı formatta mevcut

% sadece alphaları bıraktıgımız veri seti
DataSet_Alphas_Data_Order = DataSet_Master_Data_Order;
c_item_limit = size(DataSet_Master_Data_Order,2);

for i = 1:19
    DataSet_Alphas_Data_Order(1,:) = [];
end

% simdi aad, had ve pad olark 3 veri seti yapıyoruz
DataSet_Alphas_AAD_Data_Order = DataSet_Alphas_Data_Order;
DataSet_Alphas_HAD_Data_Order = DataSet_Alphas_Data_Order;
DataSet_Alphas_PAD_Data_Order = DataSet_Alphas_Data_Order;

clear DataSet_Alphas_Data_Order;

% buralarda aad, had ve pad yontemlerinde gore fake rating eklenecek olan
% user sayıları kendi matrislerine alınıyor
for i = 1:16
    DataSet_Alphas_AAD_Data_Order(9,:) = [];
end

for i = 1:8
    DataSet_Alphas_HAD_Data_Order(1,:) = [];
end

for i = 1:8
    DataSet_Alphas_HAD_Data_Order(9,:) = [];
end

for i = 1:16
    DataSet_Alphas_PAD_Data_Order(1,:) = [];
end

% aslında bu kısa bir yedekleme işlemi, direk faydası yok, kullanılmıyor
m_filename = strcat(m_dataset, "_", "data_order", ".mat");
m_filepath = strcat("../out/2_3_find_random/",m_filename);

disp(m_filepath);
save(m_filepath, 'DataSet_Master_Data_Order', '-v7.3');
% disp("export done!");

% ------------------ döngü başlıyooooooooooorrrr!!!....---------------
% projedeki en karmaşık döngü burası
for c_find_method_names = 1:c_find_method_count

    % aktif alfayı seciyoruz
    switch c_find_method_names
        case 1
            Current_Alpha_Dataset = DataSet_Alphas_AAD_Data_Order;
        case 2
            Current_Alpha_Dataset = DataSet_Alphas_HAD_Data_Order;
        case 3
            Current_Alpha_Dataset = DataSet_Alphas_PAD_Data_Order;
    end

    % aktif yaklaşımı seçiyoruz (aad, had, pad)
    m_find_method_string = m_find_method_names(c_find_method_names);
    
    % her bir alfa için 
    for c_alpha_param = 1:c_alpha_parameter_count

        m_alpha_string = sprintf('%.2f',m_alpha(c_alpha_param));

        % her bir rasgelelik için 
        for c_random = 1: m_random_count

            m_current_uir_dataset = DataSet_UIR;

            % her bir item için, yani kolon için
            for c_item = 1: c_item_limit 
                
                % eger boş ise işlem yapılmayacak, dolu ise işlem yapılacak
                % yani rasgele rating hesaplanmasını gerektirecek bir durum var mı
                % vektör dolu mu? vektör boş mu?
                % bunun için vektör toplanının sıfırdan farklı mı bakıyorum
                Current_Alpha_Dataset_Vector = Current_Alpha_Dataset(:,c_item);
                sum_Current_Alpha_Dataset_Vector = sum(Current_Alpha_Dataset_Vector);

                % eger dolu ise
                % yani fake rating eklenecek item'ı bulduk
                if (sum_Current_Alpha_Dataset_Vector > 0)

                    % aktif alfa, yani uretilecek fake rating sayısı
                    m_active_fake_rating_count = Current_Alpha_Dataset_Vector(c_alpha_param);
                        
                    % gercek veriden ilgili item vektoru alındı
                    Current_Rating_Vector = DataSet_UIR(:,c_item);
    
                    % ratinglerin ekleneceği dizi oluşturuluyor
                    m_current_index_vector =  zeros(m_user_count,2,'int32');
    
                    % diziye index ekleniyor
                    for i =1:m_user_count
                        m_current_index_vector(i,1) = i;
                    end

                    % diziye rating vermemiş userların indexleri ekleniyor
                    m_randomize_item_index = 0;
                    for i =1:m_user_count
                        m_current_rating = Current_Rating_Vector(i,1);
                        if (m_current_rating == 0)
                            m_randomize_item_index = m_randomize_item_index + 1;
                            m_current_index_vector(m_randomize_item_index, 2) = i;
                        end
                    end
    
                    %rate edilmiş itemlar oldugundan index ile user
                    %indexlerinde kayma oldu
                    %kısalma oldugundan boş olan satırları attık
                    m_current_index_vector = m_current_index_vector(1:m_randomize_item_index,:);
    
                    % random sayıları sececeğimiz vektör olusturuluyor
                    m_randomized_user_vector = zeros(m_active_fake_rating_count,1,'int32');

                    % random sayılar belirleniyor
                    % burada aynı item'a rasgele 2 kez rating eklenme
                    % girişiminin onune geçmek için while ile farklı user
                    % gelene kadar dongude tutuyoruz
                    % ---------------------------
                    m_alpha_index = 0;
                    m_randomized_user_index = 0;
                    while (m_alpha_index < m_active_fake_rating_count)
                    
                        m_while_state = true;
                        while(m_while_state)
                            m_randomized_user_index = randi([1 m_randomize_item_index]);
                    
                            m_result_randomize = find(m_randomized_user_vector==m_randomized_user_index);
                            random_is_unique = size(m_result_randomize,1);
                            
                            if (random_is_unique == 0)
                                m_alpha_index = m_alpha_index + 1;
                                m_randomized_user_vector(m_alpha_index) = m_randomized_user_index;
                                m_while_state = false;
                            else
                                m_while_state = true;
                            end
                        end
                    end
                    
                    % random vektör sıralandı
                    % yani bu userlara rating eklenecek
                    m_randomized_user_vector = sort(m_randomized_user_vector);
                    % ---------------------------

                    % burada işaretliyoruz
                    % burayı iyice kontrol et
                    % ilgili user'ın rating değeri -1 yapılıyor
                    % sonra bu -1'lere fake rating eklenecek
                    for i = 1:m_active_fake_rating_count
                        update_user_rating_index = m_randomized_user_vector(i);
                        m_current_uir_dataset(update_user_rating_index,c_item) = -1;
                    end

                    % disp("update m_current_uir_dataset");
                end

            end
            
            m_filename = strcat(m_dataset, "_", m_find_method_string, "_", m_alpha_string, "_r", int2str(c_random), ".mat");
            m_filepath = strcat("../out/2_3_find_random/",m_dataset,'/',m_filename);
            disp(m_filepath);
            save(m_filepath, 'm_current_uir_dataset', '-v7.3');
            % disp("export done!");
        
        end
        % disp("randomized done!");
    
    end
    % disp("alpha done!");

end
% disp("method done!");

disp("Finish e_2_3_find_random");

