# %%
"""
# Laboratorio No 1. Análisis Exploratorio, PCA y Apriori
"""

# %%
"""
Jose Santisteban - 21153 / Sebastian Solorzano - 21826
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# %%
"""
### 1. Exploracion de Datos
"""

# %%
df = pd.read_csv('risk_factors_cervical_cancer.csv')

print(df.info())

# %%
print(df.describe())

# %%
print(df.isnull().sum())

# %%
"""
### 6. Valores faltantes
"""

# %%
def percent_question_marks(column):
    return (column == '?').sum() / len(column) * 100

question_mark_percentages = df.apply(percent_question_marks)
columns_to_drop = question_mark_percentages[question_mark_percentages > 50].index

print("Columnas que se eliminarán:")
for col in columns_to_drop:
    print(f"{col}: {question_mark_percentages[col]:.2f}% de '?'")

df = df.drop(columns=columns_to_drop)
df = df.replace('?', np.nan)

# %%
"""
### 2. Tipo de cada variable
"""

# %%
def classify_variable(series):
    if series.dtype == 'object':
        return "Cualitativa (Categórica)"
    elif series.dtype in ['int64', 'float64']:
        if series.nunique() > 10 and (series % 1 != 0).any():
            return "Cuantitativa Continua"
        else:
            return "Cuantitativa Discreta"
    else:
        return "Tipo no reconocido"

variable_types = {col: classify_variable(df[col]) for col in df.columns}

for var, tipo in variable_types.items():
    print(f"{var}: {tipo}")

# %%
"""
### 3. Graficos exploratorios
"""

# %%
numeric_columns = df.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(nrows=len(numeric_columns)//3 + 1, ncols=3, figsize=(20, 5*len(numeric_columns)//3))
for i, column in enumerate(numeric_columns):
    sns.histplot(df[column].dropna(), ax=axes[i//3, i%3], kde=True)
    axes[i//3, i%3].set_title(column)
plt.tight_layout()
plt.show()

# %%
categorical_columns = df.select_dtypes(include=['object', 'bool']).columns
n_rows = len(categorical_columns) // 3 + (1 if len(categorical_columns) % 3 != 0 else 0)
fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(20, 5*n_rows))

if n_rows == 1:
    axes = axes.reshape(1, -1)

for i, column in enumerate(categorical_columns):
    row = i // 3
    col = i % 3
    value_counts = df[column].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[row, col])
    axes[row, col].set_title(column)
    axes[row, col].set_xlabel('')
    axes[row, col].set_ylabel('Count')
    axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
"""
### 4. Matriz de correalacion
"""

# %%
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.tight_layout()
plt.show()

# %%
"""
### 5. Tablas de frecuencia
"""

# %%
categorical_columns = df.select_dtypes(include=['object', 'bool']).columns

def freq_prop_table(df, column):
    freq = df[column].value_counts()
    prop = df[column].value_counts(normalize=True)
    table = pd.concat([freq, prop], axis=1, keys=['Frecuencia', 'Proporción'])
    table['Proporción'] = table['Proporción'].map('{:.2%}'.format)
    return table

for column in categorical_columns:
    print(f"\nTabla de Frecuencia y Proporción para {column}:")
    table = freq_prop_table(df, column)
    print(table)

n_cols = 3
n_rows = (len(categorical_columns) - 1) // n_cols + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
fig.suptitle('Distribución de Variables Categóricas', fontsize=16)

for i, column in enumerate(categorical_columns):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col] if n_rows > 1 else axes[col]
    
    sns.countplot(x=column, data=df, ax=ax)
    ax.set_title(column)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.subplots_adjust(top=0.95) 
plt.show()

# %%
"""
### 7. 
"""

# %%
def apply_one_hot_encoding(df, categorical_columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
    return pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

def apply_ordinal_encoding(df, categorical_columns):
    encoder = OrdinalEncoder()
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    return df

df_one_hot = apply_one_hot_encoding(df, categorical_columns)
df_ordinal = apply_ordinal_encoding(df.copy(), categorical_columns)

imputer = SimpleImputer(strategy='mean')
df_one_hot_imputed = pd.DataFrame(imputer.fit_transform(df_one_hot), columns=df_one_hot.columns)
df_ordinal_imputed = pd.DataFrame(imputer.fit_transform(df_ordinal), columns=df_ordinal.columns)

def perform_pca(df, title):
    pca = PCA()
    pca_result = pca.fit_transform(df)    
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title(f'Varianza Explicada Acumulada vs. Número de Componentes\n{title}')
    plt.show()
    
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"\n{title}:")
    print(f"Número de componentes para explicar el 95% de la varianza: {n_components_95}")
    print(f"Dimensionalidad original del dataset: {df.shape[1]}")

perform_pca(df[numeric_columns], "Solo Variables Numéricas")
perform_pca(df_one_hot_imputed, "Codificación One-Hot")
perform_pca(df_ordinal_imputed, "Codificación Ordinal")