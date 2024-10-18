% 2023_11_02 vesiyonu tamamlandı
% version

function [m_disguise_unif, m_disguise_norm] = generate_disguise_with_transpose(TU, betamax, sigmamax)

%disp("Start generate_disguise");

temp_DataSet = struct2cell(TU);
TU = temp_DataSet{1};
dTU = TU;

% disguise işlemini 
% userlara degil itemlara işlem yapmak için 
% matrisin transpozesini alıyorum
dTU = transpose(dTU);

% yani tüm işlemler her bir user için degil, item için yapılacak
for userid=1:size(dTU,1)
    % ratingleri 0 olmayanların indexleri bulunur
    indexOfRatings = find(dTU(userid,:)~=0);

    %indexler içindeki ratingleri aldık ve ratings diye bir vektöre koyduk    
    ratings(1,1:size(indexOfRatings,2)) = dTU(userid,indexOfRatings);    
    
    % ortalaması ve standart sapması hesaplandı
    average=mean(ratings);
    standarddeviation = std(ratings);   

    %z scoreunu için ratingten ortalamasını çıkarıp ss ye böldük
    for i=1:size(indexOfRatings,2)
        dTU(userid,indexOfRatings(1,i)) = (dTU(userid,indexOfRatings(1,i))-average)/standarddeviation;
    end
end

%disp("Finish generate_disguise");

m_disguise_unif = dTU;
m_disguise_norm = dTU;

% bu aşamada ratingler üzerine eklenecek gürültü için random sayı hesaplanacak
for userid=1:size(dTU,1)
    stddev = sigmamax * rand(1);
    delta = (betamax * rand(1))/100;
    indexOfEmptyCells = find(dTU(userid,:)==0);
    indexOfRatings = find(dTU(userid,:)~=0);
    randindex = randperm(size(indexOfEmptyCells,2));
    numberOfEmptyCellsToBeFilled = floor(delta*size(randindex,2));
    numberOfRatingsToBeDisguised = size(indexOfRatings,2);
    totalNumberOfCells = numberOfEmptyCellsToBeFilled + numberOfRatingsToBeDisguised;
        
    % unif
    alpha = sqrt(3) * stddev;
    unif_randomNumbers(1,1:totalNumberOfCells)=random('unif',-alpha,alpha,1,totalNumberOfCells);

    % norm
    norm_randomNumbers(1,1:totalNumberOfCells)=random('norm',0,stddev,1,totalNumberOfCells);
    
    indexOfRandomNumbers = 1;

    % perturbation: önce ratingler disguized edildi
    for i=1:numberOfRatingsToBeDisguised
      % dTU(userid,indexOfRatings(1,i)) = dTU(userid,indexOfRatings(1,i)) + randomNumbers(1,indexOfRandomNumbers);
        m_disguise_unif(userid,indexOfRatings(1,i)) = m_disguise_unif(userid,indexOfRatings(1,i)) + unif_randomNumbers(1,indexOfRandomNumbers);
        m_disguise_norm(userid,indexOfRatings(1,i)) = m_disguise_norm(userid,indexOfRatings(1,i)) + norm_randomNumbers(1,indexOfRandomNumbers);

        indexOfRandomNumbers = indexOfRandomNumbers + 1;
    end

    % obfuscation: sonra da boş kısımlara fake ratingler eklendi
    for i=1:numberOfEmptyCellsToBeFilled
      % dTU(userid,indexOfEmptyCells(1,randindex(1,i))) = randomNumbers(1,indexOfRandomNumbers);
        m_disguise_unif(userid,indexOfEmptyCells(1,randindex(1,i))) = unif_randomNumbers(1,indexOfRandomNumbers);
        m_disguise_norm(userid,indexOfEmptyCells(1,randindex(1,i))) = norm_randomNumbers(1,indexOfRandomNumbers);
                
        indexOfRandomNumbers = indexOfRandomNumbers + 1;
    end    
end

% itemlara disguise işlemi yapıldı
% fonksiyon çıkışında veri setini eski haline cevirmek için
% yeniden transpoze yaptık

dTU = transpose(dTU);
m_disguise_norm = transpose(m_disguise_norm);
m_disguise_unif = transpose(m_disguise_unif);

end




