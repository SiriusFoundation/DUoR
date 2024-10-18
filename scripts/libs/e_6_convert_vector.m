% 2023_11_02 vesiyonu tamamlandı

% burada tüm matrisler vektore cevriliyor, 
% sonrasında da algoritmaya sokulacak

clc;
clear;

disp("Start e_6_convert_vector");

m_dataset = 'MLM';
m_dataset = 'Yelp';
m_dataset = 'DoubanBooks';


m_source = strcat('../out/1_2_prepare_uir/',m_dataset,"/",m_dataset, '.mat');

% gercek veri okundu, okunmasının nedeni de user ve item count almak
DataSet_UIR = load(m_source);
temp_DataSet = struct2cell(DataSet_UIR);
DataSet_UIR = temp_DataSet{1};
clear temp_DataSet;

m_user_count = size(DataSet_UIR,1);
m_item_count = size(DataSet_UIR,2);

% burada diskteki tüm dosyalar sırasıyla okunacak ve 
% her dosya için 1 csv uzantılı vektör dosyası export edilecek

m_read_path = strcat("../out/5_apply_rating/",m_dataset,"/");
m_uir_file_list = dir(fullfile(m_read_path, '*.mat'));
m_file_count = size(m_uir_file_list,1);

for m_file_counter = 1:m_file_count

    m_current_file = m_uir_file_list(m_file_counter).name;
    m_current_file_with_no_extension = replace(m_current_file,".mat","");

    m_file_full_path = strcat(m_uir_file_list(m_file_counter).folder, "\",m_current_file);

    % sentetik matris dosyası okundu
    m_current_file_dataset = load(m_file_full_path);
    temp_DataSet = struct2cell(m_current_file_dataset);
    m_current_file_dataset = temp_DataSet{1};
    clear temp_DataSet;
    
    % vektör için önce rating sayısını al
    % vektörün uzunlugu için bu gerekiyor
    m_rating_count = 0;
    for i = 1:m_user_count
        for j = 1: m_item_count
            active_rating = m_current_file_dataset(i,j);
            if (active_rating ~= 0)
                m_rating_count = m_rating_count + 1;
            end
        end
    end
    
    % vektor veri yapısı olusturuldu
    m_current_vector = zeros(m_rating_count,3,"double");
    
    % sentetik matristeki hucreler geziliyor 
    % degeri 0 olmayanlar vektore satır olarak ekleniyor
    m_current_vector_index = 0;
    for i = 1:m_user_count
        for j = 1: m_item_count
            active_rating = m_current_file_dataset(i,j);
            if (active_rating ~= 0)
                
                m_current_vector_index = m_current_vector_index + 1;
                m_current_vector(m_current_vector_index,1) = i;
                m_current_vector(m_current_vector_index,2) = j;

                % bu kısımda dosyada az yer kaplasın diye 
                % round ile nokta sonrası sınırlandırıldı
                m_current_vector(m_current_vector_index,3) = round(active_rating,2);

            end
        end
    end

    % export aşaması
    m_method_string = "uir";
    m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    m_filepath = strcat("../out/6_convert_vector/", m_dataset ,"/",m_filename);

    m_message = strcat(datestr(datetime("now")), " | ", m_filepath, " | ",int2str(m_file_counter),"/",int2str(m_file_count), " | %", int2str(m_file_counter * 100 / m_file_count), " completed.");
    disp(m_message);
    m_exportFilePath_csv = strrep(m_filepath,'mat','csv');
    
    %save(m_filepath, 'm_current_vector', '-v7.3');
    writematrix(m_current_vector,m_exportFilePath_csv);

end

disp("Finish e_6_convert_vector");