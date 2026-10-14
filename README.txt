###################################
## Información sobre los Scripts ##
###################################

############################
## Generacion de Archivos ##
############################

Script 01 (01_inferir.py) genera archivos .npy y .npz:

Archivo	                                                                     |   Contenido
val_ids.npy, test_ids.npy	                                             |   ID de cada clip, en el orden del .tsv.
val_rutas.npy, test_rutas.npy	                                             |   Ruta del mp3 de cada clip.
val_y.npy, test_y.npy	                                                     |   Etiquetas reales (0/1), matriz de N clips × 50 tags.
val_pred_MTT_musicnn.npy, val_pred_MTT_vgg.npy (y los equivalentes de test)  |   Las predicciones: N clips × 50 probabilidades, una por tag.
val_parcial.npz, test_parcial.npz	                                     |   Copia de seguridad para reanudar si se corta. No se usan después.



Script 02 (02_metricas.py) genera archivos .csv:

Archivo	                                                                      |   Contenido
resumen_agregado.csv	                                                      |   Todas las métricas agregadas (macro y micro) por modelo y por tipo de umbral
por_etiqueta_MTT_musicnn_05.csv, por_etiqueta_MTT_vgg_05.csv                  |   Métricas de cada una de las 50 etiquetas con umbral 0.5
por_etiqueta_MTT_musicnn_umbral_val.csv, por_etiqueta_MTT_vgg_umbral_val.csv  |   Lo mismo, con los umbrales elegidos en validación
por_etiqueta_comparacion.csv                                                  |   Qué modelo gana en cada etiqueta según cada métrica
umbrales_elegidos_en_val.csv                                                  |   El umbral elegido para cada etiqueta y modelo



###################################
## Por qué también se define 0.5 ##
###################################

Cumple tres funciones:

* Punto de comparación: 0.5 es lo que usaría cualquiera sin ajustar nada, porque la salida es una sigmoide y se lee como probabilidad. 
Comparar 0.5 contra el umbral optimizado muestra cuánto cambia el resultado al ajustar el umbral. Eso responde parte de tu actividad 2 (trade-off Precision/Recall).

* Honestidad sobre calibración: Si 0.5 funciona mal (F1 macro de ~0.18, contra ~0.40 con umbrales optimizados), eso dice algo sobre lo calibradas que están las puntuaciones (tu actividad 4).

* Las métricas sin umbral no lo necesitan: AUC-ROC y AUC-PR no dependen de ningún umbral. Solo Precision, Recall y F1 lo necesitan.

