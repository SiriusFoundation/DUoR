clc;
clear;

disp("Start prepare debias to user item matrix");


% m_filename = "MLM.mat";
m_filename = "DoubanBooks.mat";
% m_filename = "Yelp.mat";

m_path = strcat('../output/0_datasets/', m_filename);



m_predictions = load(m_path);
temp_predictions = struct2cell(m_predictions);
m_predictions = temp_predictions{1};



% Matrisin kolon sayısını bul
numColumns = size(m_predictions, 2);

% Her kolondaki 0 olmayan elemanların sayısını tutmak için bir dizi oluştur
nonZeroCounts = zeros(1, numColumns);

% Döngü ile her kolonu kontrol et
for col = 1:numColumns
    % 0 olmayan elemanların sayısını bul
    nonZeroCounts(col) = sum(m_predictions(:, col) ~= 0);
end

% Yeni satırı oluştur (sıralı sayılarla)
newRow = 1:length(nonZeroCounts);  % Sıralı sayılar (1, 2, 3, ...)

% Yeni satırı matrisin üstüne eklemek için [newRow; nonZeroCounts] şeklinde birleştir
nonZeroCounts = [newRow; nonZeroCounts];

sortedMatrix = sortrows(nonZeroCounts', -2)';

newMatrix = sortedMatrix(1, 1:10);

disp("finish prepare debias to user item matrix");