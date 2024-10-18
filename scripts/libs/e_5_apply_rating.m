% 2023_11_02 vesiyonu tamamlandı

% bu adımda hesaplanmış olan fake ratingler ana veri setine inject edilecek
% ve yeni veri setleri generate edilecek
% 3 yaklaşım, 8 alpfa için 24 olan veri steti cesitliliği
% 7 farklı fake rating uretme yontemi ile 168 dosyalık export olusacak
% bunlar all, usr, itm alg ve disg
% ilk 3 tanesi sistem, user ve item ortalamaları olan fake ratingler,
% alg ise ilgili algoritmanın predict ettiği deger olacak
% disg ise bizim disguise ettiklerimiz

clc;
clear;

disp("Start e_5_apply_rating");

m_dataset = 'MLM';
m_dataset = 'Yelp';
m_dataset = 'DoubanBooks';


m_source = strcat("../out/1_2_prepare_uir/",m_dataset,"/",m_dataset,".mat");
m_destination = strcat("../out/5_apply_rating/",m_dataset,"/","DoubanBooks_org_0_r1_org_org.mat");

% orijinali de ekliyorum
% bunun amacı etki için ana referans kullanılacak olması
copyfile(m_source, m_destination);

% orijinal veri seti okundu
m_path = strcat('../out/1_2_prepare_uir/',m_dataset,"/",m_dataset,'.mat');

DataSet_UIR = load(m_path);
temp_DataSet = struct2cell(DataSet_UIR);
DataSet_UIR = temp_DataSet{1};

% 7 yaklaşımın fake ratingleri okunuyor
% m_path = strcat('../out/4_merge_approaches/', m_dataset ,'/', m_dataset,'_system_rating_mean.mat');
% DataSet_Average_System = load(m_path);
% temp_DataSet = struct2cell(DataSet_Average_System);
% DataSet_Average_System = temp_DataSet{1};

% m_path = strcat('../out/4_merge_approaches/', m_dataset ,'/', m_dataset,'_user_rating_mean.mat');
% Dataset_Average_User = load(m_path);
% temp_DataSet = struct2cell(Dataset_Average_User);
% Dataset_Average_User = temp_DataSet{1};

m_path = strcat('../out/4_merge_approaches/', m_dataset ,'/', m_dataset,'_item_rating_mean.mat');
Dataset_Average_Item = load(m_path);
temp_DataSet = struct2cell(Dataset_Average_Item);
Dataset_Average_Item = temp_DataSet{1};

m_path = strcat('../out/4_merge_approaches/', m_dataset ,'/', m_dataset,'_algorithm_vaecf.mat');
Dataset_Algorithm_VAECF = load(m_path);
temp_DataSet = struct2cell(Dataset_Algorithm_VAECF);
Dataset_Algorithm_VAECF = temp_DataSet{1};

% m_path = strcat('../out/4_merge_approaches/', m_dataset ,'/', m_dataset,'_disg_S2B5N.mat');
% Dataset_Disguise_S2B5N = load(m_path);
% temp_DataSet = struct2cell(Dataset_Disguise_S2B5N);
% Dataset_Disguise_S2B5N = temp_DataSet{1};

m_path = strcat('../out/4_merge_approaches/', m_dataset ,'/', m_dataset,'_disg_S3B10N.mat');
Dataset_Disguise_S3B10N = load(m_path);
temp_DataSet = struct2cell(Dataset_Disguise_S3B10N);
Dataset_Disguise_S3B10N = temp_DataSet{1};

% m_path = strcat('../out/4_merge_approaches/', m_dataset ,'/', m_dataset,'_disg_S4B25N.mat');
% Dataset_Disguise_S4B25N = load(m_path);
% temp_DataSet = struct2cell(Dataset_Disguise_S4B25N);
% Dataset_Disguise_S4B25N = temp_DataSet{1};

clear temp_DataSet;
% 4 yaklaşımın fake ratingleri okundu

% user ve item sayıları alındı
m_user_count = size(DataSet_UIR,1);
m_item_count = size(DataSet_UIR,2);

% burada diskteki tüm dosyalar sırasıyla okunacak ve 
% yani make random altındaki 7x24 tane dosya bunlar
% her dosya için 7 dosya export edilecek
m_read_file_path = strcat("../out/2_3_find_random/",m_dataset,"/");
m_uir_file_list = dir(fullfile(m_read_file_path, '*.mat'));
m_file_count = size(m_uir_file_list,1);

for m_file_counter = 1:m_file_count

    m_current_file = m_uir_file_list(m_file_counter).name;
    m_current_file_with_no_extension = replace(m_current_file,".mat","");

    m_file_full_path = strcat(m_uir_file_list(m_file_counter).folder, "\",m_current_file);

    m_current_file_dataset = load(m_file_full_path);
    temp_DataSet = struct2cell(m_current_file_dataset);
    m_current_file_dataset = temp_DataSet{1};
    clear temp_DataSet;

    % okunan 24 dosyadan ilgili olanının 7 kopyası olusturuluyor, 
    % bunlar fake ratinglerle update edilecek
    % ancak alanların -1 oldugu yerle update edilecek
    % m_current_file_dataset_smean = m_current_file_dataset;
    m_current_file_dataset_imean = m_current_file_dataset;
    % m_current_file_dataset_umean = m_current_file_dataset;
    m_current_file_dataset_algrt = m_current_file_dataset;

    % m_current_file_dataset_S2B5N = m_current_file_dataset;
    m_current_file_dataset_S3B10N = m_current_file_dataset;
    % m_current_file_dataset_S4B25N = m_current_file_dataset;

    % tum hucreler geziliyor
    for i = 1:m_user_count
        for j = 1:m_item_count

            % rating -1 mi? yani fake rating eklenecek mi?
            m_active_rating = m_current_file_dataset(i,j);
            if (m_active_rating == -1)
                
                % mean system rating
                % m_current_file_dataset_smean(i,j) = DataSet_Average_System(i,j);

                % mean user rating
                % m_current_file_dataset_umean(i,j) = Dataset_Average_User(i,j);

                % mean item rating
                m_current_file_dataset_imean(i,j) = Dataset_Average_Item(i,j);

                % algorithm
                m_current_file_dataset_algrt(i,j) = Dataset_Algorithm_VAECF(i,j);
                
                % disguise
                % m_current_file_dataset_S2B5N(i,j) = Dataset_Disguise_S2B5N(i,j);
                m_current_file_dataset_S3B10N(i,j) = Dataset_Disguise_S3B10N(i,j);
                % m_current_file_dataset_S4B25N(i,j) = Dataset_Disguise_S4B25N(i,j);
               
            end
        end
    end

    % üretilenler
    % m_method_string = "avg_all";
    % m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    % m_filepath = strcat("../out/5_apply_rating/",m_dataset,"/",m_filename);
    % disp(m_filepath);
    % save(m_filepath, 'm_current_file_dataset_smean', '-v7.3');

    % m_method_string = "avg_usr";
    % m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    % m_filepath = strcat("../out/5_apply_rating/",m_dataset,"/",m_filename);
    % disp(m_filepath);
    % save(m_filepath, 'm_current_file_dataset_umean', '-v7.3');

    m_method_string = "avg_itm";
    m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    m_filepath = strcat("../out/5_apply_rating/",m_dataset,"/",m_filename);
    disp(m_filepath);
    save(m_filepath, 'm_current_file_dataset_imean', '-v7.3');

    m_method_string = "alg_vaecf";
    m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    m_filepath = strcat("../out/5_apply_rating/",m_dataset,"/",m_filename);
    disp(m_filepath);
    save(m_filepath, 'm_current_file_dataset_algrt', '-v7.3');

    % m_method_string = "xdisg_S2B5N";
    % m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    % m_filepath = strcat("../out/5_apply_rating/",m_dataset,"/",m_filename);
    % disp(m_filepath);
    % save(m_filepath, 'm_current_file_dataset_S2B5N', '-v7.3');

    m_method_string = "xdisg_S3B10N";
    m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    m_filepath = strcat("../out/5_apply_rating/",m_dataset,"/",m_filename);
    disp(m_filepath);
    save(m_filepath, 'm_current_file_dataset_S3B10N', '-v7.3');

    % m_method_string = "xdisg_S4B25N";
    % m_filename = strcat(m_current_file_with_no_extension, "_", m_method_string, ".mat");
    % m_filepath = strcat("../out/5_apply_rating/",m_dataset,"/",m_filename);
    % disp(m_filepath);
    % save(m_filepath, 'm_current_file_dataset_S4B25N', '-v7.3');
    
end

disp("Finish e_5_apply_rating");