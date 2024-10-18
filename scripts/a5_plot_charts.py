import pandas as pd
import matplotlib.pyplot as plt

print("a5_plot_charts start")

m_dataset = 'MLM'
file_path = '..\\output\\4_evaluations\\mlm_all.csv'  # CSV dosyasının yolunu buraya yaz


data_table = pd.read_csv(file_path, sep=';')

data_table['Debiasing'] = 'new_data'
data_table['Dataset'] = 'new_data'
data_table['Algorithm'] = 'new_data'

for index, row in data_table.iterrows():
    filename = row['filename']
    split_values = filename.split('_')
    remove_ext = split_values[2]

    temp_string = remove_ext.split('.')
    temp_string = temp_string[0]

    data_table.at[index, 'Debiasing'] = split_values[0]
    data_table.at[index, 'Dataset'] = split_values[1]
    data_table.at[index, 'Algorithm'] = temp_string

data_table = data_table.drop(['filename', 'date'], axis=1)
new_columns_order = ['counter', 'Dataset', 'Algorithm', 'Debiasing', 'BTA', 'APRI', 'RMSE-PC', 'NDCG', 'Precision', 'Recall', 'F1', 'APLT', 'Novelty', 'MRM', 'LTC', 'Entropy']
data_table = data_table[new_columns_order]

data_table_HPF = data_table[data_table['Algorithm'] == 'HPF']
data_table_SKM = data_table[data_table['Algorithm'] == 'SKM']

data_table_HPF = data_table.drop(['counter', 'Dataset', 'Algorithm'], axis=1)
data_table_SKM = data_table.drop(['counter', 'Dataset', 'Algorithm'], axis=1)

selected_columns = ['Debiasing', 'NDCG']
data_table_HPF_plot = data_table_HPF[selected_columns]
data_table_SKM_plot = data_table_SKM[selected_columns]

df = data_table_HPF_plot

plt.figure(figsize=(10, 6))  # Grafik boyutunu ayarlama
plt.bar(df['NDCG'], df['Debiasing'], color='skyblue')

# Grafik başlık ve etiketleri
plt.xlabel('Debiasing')
plt.ylabel('NDCG')
plt.title('MLM - NDCG')


write_file_path = '..\\output\\6_plot_drawing\\bar_chart_output.pdf'
plt.savefig(write_file_path, format='pdf')

# Grafiği gösterme (eğer sadece PDF kaydedip göstermek istemezsen bunu atlayabilirsin)
plt.show()


print("a5_plot_charts finish")