# 📊 Análisis del Mercado Laboral Argentino (EPH 2016–2025)

Trabajo práctico realizado para la materia **Introducción al Análisis de Datos**, utilizando microdatos de la  
**Encuesta Permanente de Hogares (EPH – INDEC)** para el período **2016–2025**.

El proyecto analiza la evolución del mercado laboral y los ingresos reales en dos aglomerados:
- **Posadas**
- **Comodoro Rivadavia – Rada Tilly**

El foco principal del trabajo está puesto en el **desarrollo de un modelo de Árbol de Decisión** para la imputación de ingresos no declarados, priorizando la interpretabilidad del modelo.

---

## 🎯 Objetivos del proyecto

- Analizar la evolución de:
  - Tasa de actividad
  - Tasa de empleo
  - Tasa de desocupación
  - Ingresos nominales y reales (ajustados por inflación)
- Comparar dinámicas laborales entre dos aglomerados con estructuras económicas distintas
- Aplicar análisis univariado y multivariado
- Desarrollar un **modelo estadístico interpretable** para imputar ingresos faltantes
- Visualizar información mediante gráficos y mapas georreferenciados

---

## 🗂️ Fuente de datos

- **Encuesta Permanente de Hogares (EPH)** – INDEC  
- Período: **2016–2025 (todos los trimestres disponibles)**
- Ingresos deflactados a **pesos constantes del 2° trimestre de 2025** utilizando el IPC trimestral

---

## 🧹 Procesamiento y limpieza de datos

- Integración de bases de personas en un único dataset
- Población objetivo: personas **ocupadas mayores de 18 años**
- Eliminación de:
  - Edades inválidas
  - Ingresos negativos o nulos
  - Ponderadores faltantes
  - Registros duplicados
- Construcción de variables derivadas:
  - Años de educación
  - Edad al cuadrado
  - Formalidad laboral
- Estandarización de variables sociodemográficas y laborales

---

## 📈 Análisis exploratorio

El análisis descriptivo muestra diferencias estructurales claras entre ambos aglomerados:

- **Posadas**
  - Mercado laboral más estable
  - Brechas de género moderadas
  - Caída sostenida del ingreso real desde 2017

- **Comodoro Rivadavia – Rada Tilly**
  - Mayores niveles de actividad e ingresos
  - Dinámica más volátil
  - Brechas de género y salariales más marcadas

Se analizaron medidas de tendencia central, percentiles, correlaciones, tasas por sexo, edad y categoría ocupacional, así como visualizaciones temporales y geográficas.

---

## 🌳 Modelo de Árbol de Decisión – Predicción del Ingreso Real

### 🔍 Objetivo del modelo
Imputar **ingresos laborales reales no declarados** manteniendo coherencia económica e interpretabilidad.

### 🧠 Modelo utilizado
- **Árbol de Decisión (Decision Tree Regressor)**

Se eligió este modelo por su capacidad de:
- Identificar relaciones no lineales
- Adaptarse a estructuras laborales distintas
- Permitir una **lectura clara de las reglas de decisión**

### 📥 Variables incluidas
- Edad
- Nivel educativo
- Formalidad laboral
- Estado civil
- Horas trabajadas
- Variables derivadas (edad², años de educación)

---

## 📊 Evaluación del modelo

El rendimiento se evaluó mediante:
- **MAE (Error Absoluto Medio)**
- **RMSE**
- **R²**

Los resultados muestran que el modelo:
- Captura buena parte de la estructura del ingreso real
- Presenta diferencias según el aglomerado
- Es lo suficientemente sólido para imputar ingresos faltantes sin perder coherencia interna

---

## 🧩 Resultados clave del Árbol

### 📍 Posadas
- Variable más relevante: **formalidad laboral**
- Informales con pocas horas (<17 hs): ingresos muy bajos (~$150.000)
- A mayor carga horaria, edad y educación → ingresos entre $380.000 y $520.000
- Formales con mayor carga horaria y educación alta → hasta **$1.29 millones**

👉 Regla clara: **formalidad + horas + educación** explican el ingreso.

---

### 📍 Comodoro Rivadavia – Rada Tilly
- Ingresos estructuralmente más altos
- Informales con pocas horas: ~$275.000
- Formales:
  - Hasta 31 hs: ~$1.44 millones
  - Jornadas largas + educación media: $1.56–$1.96 millones
  - Niveles más altos: hasta **$2.6 millones**

👉 En este aglomerado cobran más peso:
- Edad
- Intensidad de la jornada
- Estabilidad familiar

---

## 🧠 Interpretación del modelo

El Árbol de Decisión **no replica una única lógica** para todos los casos:
- Se adapta automáticamente a la estructura laboral de cada ciudad
- Selecciona variables relevantes según el contexto local
- Permite explicar con claridad **por qué** se asigna determinado ingreso

Esto hace que las imputaciones sean **consistentes, interpretables y realistas**.

---

## 🧾 Conclusiones

- Existen dos mercados laborales con dinámicas muy distintas
- La brecha de ingresos reales entre Posadas y Comodoro es amplia y persistente
- La formalidad, las horas trabajadas, la educación y la edad son los principales determinantes del ingreso
- El Árbol de Decisión resulta una herramienta adecuada para imputar ingresos sin perder trazabilidad analítica

---

## 👥 Trabajo en equipo

Proyecto realizado en grupo para la materia **Introducción al Análisis de Datos**.

---

## 🛠️ Tecnologías utilizadas
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Datos oficiales INDEC (EPH)
