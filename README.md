# vrmachine


datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()



datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()



X = datos.loc[:, datos.columns != 'BuenPagador']
X


y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)


preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)


modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVC(kernel="rbf", gamma = 5, C = 10))
])

modelo.fit(X_train, y_train)


pred = modelo.predict(X_test)
pred


labels = ["Si", "No"]
MC = confusion_matrix(y_test, pred, labels=labels)
MC


indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))


  print("\nComparación Final")


resultados = {
    "Random Forest": metricas_rf,
    "AdaBoost": metricas_ada,
    "Gradient Boosting": metricas_gb,
    "XGBoost": metricas_xgb,
    "Árbol de Decisión": metricas_dt
}


comparacion = pd.DataFrame(columns=[
    "Precisión Global", "Error Global", "Precisión Positiva (PP)", "Precisión Negativa (PN)"
])

for nombre, met in resultados.items():
    comparacion.loc[nombre] = [
        met["Precisión Global"], met["Error Global"],
        met["Precisión Positiva (PP)"], met["Precisión Negativa (PN)"]
    ]

comparacion = comparacion.sort_values(by="Precisión Global", ascending=False)
print("\nTabla comparativa de modelos:")
print(comparacion)
