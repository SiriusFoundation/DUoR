% burağın hazırladığı veri alınarak matrise cevrilecek


clc;
clear;

disp("Start prepare debias to user item matrix");

m_dataset = "MLM";m_algorithm = "SKM";m_debias_name = "discreapancy_minimization";
m_dataset = "MLM";m_algorithm = "SKM";m_debias_name = "fair_recommendations";
m_dataset = "MLM";m_algorithm = "HPF";m_debias_name = "discreapancy_minimization";
m_dataset = "MLM";m_algorithm = "HPF";m_debias_name = "fair_recommendations";

m_dataset = "DB";m_algorithm = "SKM";m_debias_name = "discreapancy_minimization";
m_dataset = "DB";m_algorithm = "SKM";m_debias_name = "fair_recommendations";
m_dataset = "DB";m_algorithm = "HPF";m_debias_name = "discreapancy_minimization";
m_dataset = "DB";m_algorithm = "HPF";m_debias_name = "fair_recommendations";

m_dataset = "YL";m_algorithm = "SKM";m_debias_name = "discreapancy_minimization";
m_dataset = "YL";m_algorithm = "SKM";m_debias_name = "fair_recommendations";
m_dataset = "YL";m_algorithm = "HPF";m_debias_name = "discreapancy_minimization";
m_dataset = "YL";m_algorithm = "HPF";m_debias_name = "fair_recommendations";

% m_dataset = "DB";
% m_dataset = "YL";
% m_algorithm = "SKM";
% m_algorithm = "HPF";
% m_debias_name = "discreapancy_minimization";
% m_debias_name = "fair_recommendations";


m_filename = strcat(m_debias_name, '_', m_dataset, '_', m_algorithm);
m_path = strcat('../output/1_prepare/', m_filename);
m_save_path = strcat('../output/3_debiasing/',m_filename);
m_save_path_csv = strcat(m_save_path, '.csv');
m_save_path_mat = strcat(m_save_path, '.mat');


m_topn_vector = readmatrix(m_path);

[m_topn_vector_size, ~] = size(m_topn_vector);
m_user_count = max(m_topn_vector(:, 1));
m_topn_matrix = zeros(m_user_count, 10);

for i = 1:m_user_count
    
    filteredMatrix = m_topn_vector(m_topn_vector(:, 1) == i, :);

    if ~isempty(filteredMatrix)
        m_topn_matrix(i,1) = filteredMatrix(1, 2);
        m_topn_matrix(i,2) = filteredMatrix(2, 2);
        m_topn_matrix(i,3) = filteredMatrix(3, 2);
        m_topn_matrix(i,4) = filteredMatrix(4, 2);
        m_topn_matrix(i,5) = filteredMatrix(5, 2);
        m_topn_matrix(i,6) = filteredMatrix(6, 2);
        m_topn_matrix(i,7) = filteredMatrix(7, 2);
        m_topn_matrix(i,8) = filteredMatrix(8, 2);
        m_topn_matrix(i,9) = filteredMatrix(9, 2);
        m_topn_matrix(i,10) = filteredMatrix(10, 2);
    else
        % mlm'de 2909 yok
        % yelp 7806 yok
        % db 1375, 1377 yok
        disp(strcat(m_dataset, " - ", m_algorithm, " - ", m_debias_name, ": ", num2str(i), " indisinde hata var!"));

        if strcmp(m_dataset, 'MLM')
            m_topn_matrix(i, :) = [2858, 260, 1196, 1210, 480, 2028, 589, 2571, 1270, 593];
        elseif strcmp(m_dataset, 'DB')
            m_topn_matrix(i, :) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        elseif strcmp(m_dataset, 'YL')
            m_topn_matrix(i, :) = [10275, 7414, 4790, 3567, 6679, 11661, 7290, 7608, 209, 12090];
        else
            disp('Tanımsız değer');
end


    end

%   if i == 5000
%       disp(i);
%   end
end


% CSV dosyasına kaydetme
writematrix(m_topn_matrix, m_save_path_csv);

% MATLAB dosyasına kaydetme
save(m_save_path_mat, 'm_topn_matrix');


disp(strcat(m_dataset, " - ", m_algorithm, " - ", m_debias_name));
disp("Finish prepare debias to user item matrix");
