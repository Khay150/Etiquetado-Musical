"""
02_metricas.py
Compara MTT_musicnn vs MTT_vgg con AUC-ROC, AUC-PR, Precision, Recall y F1,
en forma agregada (macro / micro) y etiqueta por etiqueta, y muestra si cambia el "ganador".

Uso:
    python 02_metricas.py --root "D:/.../Etiquetado-Musical"
Lee  <root>/resultados/*.npy (salida de 01_inferir.py) y escribe CSVs en <root>/resultados/.
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODELOS = ["MTT_musicnn", "MTT_vgg"]


def cargar(out, split):
    y = np.load(os.path.join(out, f"{split}_y.npy"))
    preds = {m: np.load(os.path.join(out, f"{split}_pred_{m}.npy")) for m in MODELOS}
    ok = np.all([~np.isnan(p).any(axis=1) for p in preds.values()], axis=0)  # descarta clips con error
    if (~ok).any():
        print(f"[{split}] aviso: {(~ok).sum()} clips sin prediccion, se excluyen")
    return y[ok], {m: p[ok] for m, p in preds.items()}


def prf(y, pred_bin):
    """Precision, recall y F1 por columna (con 0 si el denominador es 0)."""
    tp = (y * pred_bin).sum(0).astype(float)
    pp = pred_bin.sum(0).astype(float)
    pos = y.sum(0).astype(float)
    p = np.divide(tp, pp, out=np.zeros_like(tp), where=pp > 0)
    r = np.divide(tp, pos, out=np.zeros_like(tp), where=pos > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f


def prf_micro(y, pred_bin):
    tp = (y * pred_bin).sum()
    p = tp / max(pred_bin.sum(), 1)
    r = tp / max(y.sum(), 1)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def umbral_optimo_f1(y, s):
    """Umbral por etiqueta que maximiza F1 (se elige en validacion)."""
    grilla = np.round(np.linspace(0.01, 0.99, 99), 2)
    umb = np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        f1s = [prf(y[:, [j]], (s[:, [j]] >= t).astype(int))[2][0] for t in grilla]
        umb[j] = grilla[int(np.argmax(f1s))]
    return umb


def metricas(y, s, umbral):
    """Devuelve (dict agregado, DataFrame por etiqueta) para un modelo y un umbral (escalar o vector)."""
    pred = (s >= umbral).astype(int)
    p, r, f = prf(y, pred)
    pm, rm, fm = prf_micro(y, pred)
    roc = np.array([roc_auc_score(y[:, j], s[:, j]) for j in range(y.shape[1])])
    ap = np.array([average_precision_score(y[:, j], s[:, j]) for j in range(y.shape[1])])
    agg = {
        "AUC-ROC macro": roc.mean(), "AUC-ROC micro": roc_auc_score(y.ravel(), s.ravel()),
        "AUC-PR macro": ap.mean(), "AUC-PR micro": average_precision_score(y.ravel(), s.ravel()),
        "Precision macro": p.mean(), "Precision micro": pm,
        "Recall macro": r.mean(), "Recall micro": rm,
        "F1 macro": f.mean(), "F1 micro": fm,
    }
    por_tag = pd.DataFrame({"AUC-ROC": roc, "AUC-PR": ap, "Precision": p, "Recall": r, "F1": f})
    return agg, por_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    out = os.path.join(os.path.abspath(args.root), "resultados")

    with open(os.path.join(args.root, "output_labels_mtt.txt"), encoding="utf8") as f:
        etiquetas = ast.literal_eval(f.read().split("=", 1)[1].strip())

    yv, pv = cargar(out, "val")
    yt, pt = cargar(out, "test")
    print(f"val: {len(yv)} clips | test: {len(yt)} clips")

    # Umbrales por etiqueta elegidos SOLO en validacion; se aplican a test.
    umb_val = {m: umbral_optimo_f1(yv, pv[m]) for m in MODELOS}

    configuraciones = {
        "umbral 0.5": {m: 0.5 for m in MODELOS},
        "umbral por etiqueta (elegido en val)": umb_val,
    }

    filas, por_tag_all = [], {}
    for cfg_nombre, umbrales in configuraciones.items():
        for m in MODELOS:
            agg, tag = metricas(yt, pt[m], umbrales[m])
            filas.append({"modelo": m, "umbral": cfg_nombre, **agg})
            tag.insert(0, "etiqueta", etiquetas)
            tag["positivos_test"] = yt.sum(0)
            por_tag_all[(cfg_nombre, m)] = tag
    resumen = pd.DataFrame(filas)
    resumen.to_csv(os.path.join(out, "resumen_agregado.csv"), index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format)

    print("\n=== METRICAS AGREGADAS EN TEST ===")
    for cfg_nombre in configuraciones:
        print(f"\n-- {cfg_nombre} --")
        sub = resumen[resumen.umbral == cfg_nombre].set_index("modelo").drop(columns="umbral").T
        sub["ganador"] = sub.idxmax(axis=1)
        print(sub)

    print("\n=== ETIQUETA POR ETIQUETA (umbral 0.5): cuantas etiquetas gana cada modelo ===")
    a = por_tag_all[("umbral 0.5", MODELOS[0])]
    b = por_tag_all[("umbral 0.5", MODELOS[1])]
    tabla = []
    for met in ["AUC-ROC", "AUC-PR", "Precision", "Recall", "F1"]:
        gana_a = (a[met] > b[met]).sum()
        gana_b = (b[met] > a[met]).sum()
        tabla.append({"metrica": met, f"gana {MODELOS[0]}": gana_a, f"gana {MODELOS[1]}": gana_b,
                      "empates": 50 - gana_a - gana_b})
    print(pd.DataFrame(tabla).to_string(index=False))

    # Detalle por etiqueta: ganador segun cada metrica y si coincide entre metricas
    det = a[["etiqueta", "positivos_test"]].copy()
    for met in ["AUC-ROC", "AUC-PR", "F1"]:
        det[f"dif {met} (musicnn-vgg)"] = a[met] - b[met]
        det[f"gana {met}"] = np.where(a[met] > b[met], MODELOS[0], np.where(b[met] > a[met], MODELOS[1], "empate"))
    # "cambia" solo si dos metricas senalan ganadores DISTINTOS (los empates, p. ej. F1=0 en ambos, no cuentan)
    det["ganador cambia segun metrica"] = det[["gana AUC-ROC", "gana AUC-PR", "gana F1"]].apply(
        lambda fila: len(set(fila) - {"empate"}) > 1, axis=1)
    det.sort_values("positivos_test", ascending=False).to_csv(
        os.path.join(out, "por_etiqueta_comparacion.csv"), index=False)
    print(f"\nEtiquetas donde el ganador cambia segun la metrica (ROC / PR / F1): "
          f"{int(det['ganador cambia segun metrica'].sum())} de 50")
    print(det[det["ganador cambia segun metrica"]][["etiqueta", "positivos_test", "gana AUC-ROC", "gana AUC-PR", "gana F1"]]
          .to_string(index=False))

    for (cfg_nombre, m), tag in por_tag_all.items():
        sufijo = "05" if cfg_nombre == "umbral 0.5" else "umbral_val"
        tag.to_csv(os.path.join(out, f"por_etiqueta_{m}_{sufijo}.csv"), index=False)
    pd.DataFrame({"etiqueta": etiquetas, **{f"umbral_{m}": umb_val[m] for m in MODELOS}}).to_csv(
        os.path.join(out, "umbrales_elegidos_en_val.csv"), index=False)
    print(f"\nCSVs guardados en {out}")


if __name__ == "__main__":
    main()
