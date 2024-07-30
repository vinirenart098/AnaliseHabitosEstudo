
# AnaliseHabitosEstudo

## Descrição

Projeto de análise de hábitos de estudo usando reconhecimento de padrões, redes neurais e aprendizado de máquina. Utilizando Python e TensorFlow, treinamos um modelo para prever hábitos de estudo com base em dados coletados, identificando padrões e fatores que influenciam a eficácia dos estudos. - Formulário https://docs.google.com/forms/d/e/1FAIpQLSfbTKa68HnNHK86pON00KLN7AD7sqdrOebFPvy_Rf2mWbtfUQ/viewform

## Estrutura do Projeto

- `data/`: Contém os dados coletados via Google Forms.
- `notebooks/`: Jupyter Notebooks com o código de análise e treinamento do modelo.
- `models/`: Modelos treinados e salvos.
- `scripts/`: Scripts Python para pré-processamento de dados e treinamento do modelo.
- `README.md`: Este arquivo.

## Pré-requisitos

- Python 3.7+
- TensorFlow
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Instalação

Clone o repositório e instale os pacotes necessários:

```bash
git clone https://github.com/seu-usuario/AnaliseHabitosEstudo.git
cd AnaliseHabitosEstudo
pip install -r requirements.txt
```

## Uso

1. **Carregar e Pré-processar Dados:**

   No notebook `notebooks/data_preprocessing.ipynb`, carregue os dados do Google Forms e pré-processe-os para análise.

2. **Análise Exploratória dos Dados:**

   Use `notebooks/data_analysis.ipynb` para visualizar e analisar os padrões nos dados.

3. **Treinamento do Modelo:**

   No notebook `notebooks/model_training.ipynb`, treine o modelo de rede neural com TensorFlow.

4. **Fazer Previsões:**

   Utilize `notebooks/model_prediction.ipynb` para fazer previsões com novos dados.

## Exemplo de Código

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Carregar dados
df = pd.read_excel('data/Respostas_de_habitos.xlsx')
df.columns = df.columns.str.strip()

# Pré-processamento
label_encoders = {}
for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Dividir dados em X e y
X = df.drop('HorasEstudo', axis=1)
y = df['HorasEstudo']

# Dividir em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir o modelo
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(y.unique()), activation='softmax')
])

# Compilar e treinar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Fazer previsões
new_data = pd.DataFrame({
    'HorarioEstudo': [label_encoders['HorarioEstudo'].transform(['Manhã'])[0]],
    'LocalEstudo': [label_encoders['LocalEstudo'].transform(['Em casa (quarto, sala, etc.)'])[0]],
    'MetodosEstudo': [label_encoders['MetodosEstudo'].transform(['Leituras de livros e artigos'])[0]],
    'FazPausas': [label_encoders['FazPausas'].transform(['Sim'])[0]],
    'FrequenciaPausas': [label_encoders['FrequenciaPausas'].transform(['A cada 1 hora'])[0]],
    'MotivacaoEstudo': [label_encoders['MotivacaoEstudo'].transform(['Interesse pelo assunto'])[0]],
    'DistracoesEstudo': [label_encoders['DistracoesEstudo'].transform(['Redes sociais'])[0]],
    'FerramentasEstudo': [label_encoders['FerramentasEstudo'].transform(['Sites educacionais (Artigos) / Alura - Udemy - Brasil Escola'])[0]],
    'EficaciaMetodos': [label_encoders['EficaciaMetodos'].transform(['Eficaz'])[0]],
    'MelhoraResultados': [label_encoders['MelhoraResultados'].transform(['Sim'])[0]]
})
prediction = model.predict(new_data)
predicted_hours = label_encoders['HorasEstudo'].inverse_transform([prediction.argmax()])[0]
print(f'Predicted Hours of Study: {predicted_hours}')
```

## Contribuição

Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
