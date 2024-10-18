% version
% generate_uir.m
% v8.2_29.09.2022
% notes:
%

function uir = generate_uir(DataSet)

    % veriyi okuyunca struct veri yapısında alıyor, aşağıdaki blok ile
    % struct veri yapısının içinden matris alınıyor.
    temp_DataSet = struct2cell(DataSet);
    DataSet = temp_DataSet{1};

    % rating sayısı, döngüde kullanılacak
    observation_count = size(DataSet,1);
    
    % user ve item vektörü oluşturuldu
    user_vector = DataSet(:,1);
    item_vector = DataSet(:,2); 
    % rating_vector = DataSet(:,3);
    
    % bu kısım doğru mu? max değerleri user ve item sayısı olarak
    % yorumladım
    user_count = max(user_vector);
    item_count = max(item_vector);

    %user_count_2 = (unique(user_vector));
    %item_count_2 = (unique(item_vector));

    user_count = size(unique(user_vector),1);
    item_count = size(unique(item_vector),1);
    % 3952 - 3706 = 246
    
    % boş matris olusturuldu
    uir = zeros(user_count,item_count,'uint32');
    uir = double(uir);
    
    % her bir rating için
    for i=1:observation_count
        user = DataSet(i,1);
        item = DataSet(i,2);
        rating = DataSet(i,3);
        
        % rating matrise yerleştiriliyor
        uir(user,item) = rating;
    end
end




