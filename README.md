# Telecom X - Predicción de Cancelación de Clientes (Churn)

## 📋 Descripción del Proyecto
Análisis predictivo para identificar clientes con mayor probabilidad de cancelar sus servicios en Telecom X. Este proyecto forma parte del Challenge de Data Science de Oracle Next Education (ONE) y Alura Latam.

## 🎯 Objetivo
Desarrollar modelos de machine learning que permitan anticipar la cancelación de clientes y proporcionar insights estratégicos para retención.

## 📊 Dataset
- **Fuente**: Datos procesados de la Parte 1 del desafío
- **Registros**: 7,043 clientes
- **Variables**: Información demográfica, servicios contratados, cargos y antigüedad
- **Variable objetivo**: Churn (0 = No canceló, 1 = Canceló)

## 🛠️ Tecnologías Utilizadas
- Python 3.8+
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

## 📁 Estructura del Proyecto

```bash
TelecomX-part2/
├── TelecomX_Part2.ipynb # Notebook principal
├── telecomx_data_tratado.csv # Dataset procesado
└── README.md # Documentación
```

## 📈 Metodología

### Fase 1: Preparación de Datos
```python
# Carga y limpieza
df = pd.read_csv('telecomx_data_tratado.csv')
df = df.drop('customerID', axis=1)

# Encoding one-hot
df = pd.get_dummies(df, columns=df.select_dtypes(['object']).columns, drop_first=True)

# Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Estandarización
scaler = StandardScaler()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_res[numeric_cols])
```

### Fase 2: Análisis Exploratorio
1. Matriz de correlación
2. Análisis dirigido de variables clave:
- Antigüedad vs Churn
- Gasto total vs Churn
- Tipo de contrato vs Churn

### Fase 3: Modelado Predictivo

#### Modelo 1: Regresión Logística
```python
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train_res)
```

#### Modelo 2: Random Forest
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)
```

### Fase 4: Evaluación
- Matrices de confusión
- Classification report
- Curvas ROC y AUC

## 🏆 Resultados

### Mejor Modelo: Random Forest

```bash
Métrica	            Regresión Logística	      Random Forest
AUC	                    0.843	                  0.851
Precisión (Churn)	    0.657	                  0.682
Recall (Churn)	        0.553	                  0.551
F1-Score (Churn)	    0.600	                  0.610
```

### Top 5 Variables Más Importantes
1. antiguedad_meses - Antigüedad del cliente
2. cargos_totales - Gasto total acumulado
3. tipo_contrato_Month-to-month - Contrato mensual
4. cargos_mensuales - Gasto mensual
5. servicio_internet_Fiber optic - Fibra óptica

## 🔍 Insights Clave

Perfil de Cliente con Alto Riesgo

```bash
Característica	          Valor de Riesgo
Contrato	              Mensual (Month-to-month)
Antigüedad	              Menos de 12 meses
Servicio Internet	      Fibra óptica sin soporte técnico
Cargos mensuales	      Superiores a $70
```

### Factores Protectores
```bash
✅ Contratos de 1 o 2 años
✅ Mayor antigüedad (+24 meses)
✅ Servicios adicionales contratados (soporte, seguridad)
```

## 💡 Estrategias de Retención

Corto Plazo
```bash
🚨 Alertas tempranas para perfiles de riesgo
🎯 Ofertas de conversión a contratos anuales
📱 Campañas enfocadas en primeros 12 meses
```

Mediano Plazo
```bash
📦 Bundles con soporte técnico incluido
💰 Descuentos por fidelidad
📊 Encuestas de satisfacción post-compra
```

Largo Plazo
```bash
🤖 Integración del modelo predictivo al CRM
🎨 Personalización de ofertas por perfil
🏆 Programa de beneficios escalonados
```

## 📊 Visualizaciones Incluidas
- Distribución de Churn
- Matriz de correlación
- Boxplots (antigüedad y gasto vs Churn)
- Barras apiladas (contrato y servicio vs Churn)
- Matrices de confusión
- Curvas ROC comparativas
- Importancia de variables

## 🚀 Cómo Ejecutar

### Opción 1: Google Colab
https://colab.research.google.com/assets/colab-badge.svg

### Opción 2: Local
```bash
# Clonar repositorio
git clone https://github.com/AllanBOG/TelecomX-part2.git
```

# Instalar dependencias
```python
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

# Ejecutar Jupyter Notebook
jupyter notebook TelecomX_Part2.ipynb

## 📦 Instalación Rápida
```bash
pip install -r requirements.txt
```

## 📋 requirements.txt

```bash
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
imbalanced-learn==0.10.1
jupyter==1.0.0
```

## 🏅 Autor
**Allan BOG**  
Data Science Student | Oracle Next Education

## 📄 Licencia
Este proyecto fue desarrollado con fines educativos como parte del programa Oracle Next Education (ONE) en colaboración con Alura Latam.
