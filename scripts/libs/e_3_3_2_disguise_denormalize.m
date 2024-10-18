% 2023_11_02 vesiyonu tamamlandı
% denormalizasyon işlemi, burada sorun olabilir

clc;
clear;

disp("Start e_3_3_2_disguise_denormalize");

m_dataset = "MLM";
m_dataset = "Yelp";
m_dataset = "DoubanBooks";

m_path = '../out/1_2_prepare_uir/';
m_path = strcat(m_path,m_dataset,"/",m_dataset,".mat");

m_dataSet_original = load(m_path);
temp_DataSet = struct2cell(m_dataSet_original);
m_dataSet_original = temp_DataSet{1};





m_read_file_path = strcat("../out/3_3_1_disguise_approach/",m_dataset,"/");
m_disguise_file_list = dir(fullfile(m_read_file_path, '*.mat'));
m_file_count = size(m_disguise_file_list,1);

for m_file_counter = 1:m_file_count

    m_current_file = m_disguise_file_list(m_file_counter).name;
    m_current_file_with_no_extension = replace(m_current_file,".mat","");

    m_file_full_path = strcat(m_disguise_file_list(m_file_counter).folder, "\",m_current_file);

    m_current_file_dataset = load(m_file_full_path);
    temp_DataSet = struct2cell(m_current_file_dataset);
    m_current_file_dataset = temp_DataSet{1};
    clear temp_DataSet;

    % phase 2'de userlara degil itemlara işlem yapmak için matrisin
    % transpozesini alıyorum
    dTU = transpose(m_current_file_dataset);
    m_dataSet_original = transpose(m_dataSet_original);
    
    % yani tüm işlemler her bir user için yapılacak
    for userid=1:size(dTU,1)
        % ratingleri 0 olmayanların indexleri bulunur
        % indexOfRatings = find(dTU(userid,:)~=0);
        indexOfRatings = find(m_dataSet_original(userid,:)~=0);
    
        %indexler içindeki ratingleri aldık ve ratings diye bir vektöre koyduk    
        % ratings(1,1:size(indexOfRatings,2)) = dTU(userid,indexOfRatings);    
        ratings(1,1:size(indexOfRatings,2)) = m_dataSet_original(userid,indexOfRatings);    
        
        % ortalaması ve standart sapması hesaplandı
        average = mean(ratings);
        standarddeviation = std(ratings);   
    
        %z scoreunu geri almak için ratingi ss ile çarpıp ortalamasını ekledim
        for i=1:size(indexOfRatings,2)

            dTU(userid,indexOfRatings(1,i)) = (dTU(userid,indexOfRatings(1,i)) * standarddeviation) + average;
        
        end
    end

    dTU = transpose(dTU);
    m_dataSet_original = transpose(m_dataSet_original);

    m_param_list = split(m_current_file_with_no_extension,"_");
    m_method_string = strcat("disg","_",m_param_list(2),m_param_list(3),m_param_list(4));
    m_filename = strcat(m_dataset, "_", m_method_string, ".mat");
    m_filepath = strcat("../out/3_3_2_disguise_denormalize/",m_dataset,"/",m_filename);
    save(m_filepath, 'dTU', '-v7.3');
    m_filepath = strcat(m_filepath, " - ", int2str(m_file_counter), "/", int2str(m_file_count));
    disp(m_filepath);
end

disp("Finish e_3_3_2_disguise_denormalize");