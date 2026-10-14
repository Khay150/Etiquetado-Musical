"""
01_inferir.py
Corre MTT_musicnn y MTT_vgg sobre validacion (carpeta c) y evaluacion (d, e, f) de MagnaTagATune
y guarda las puntuaciones por clip (vector de 50 probabilidades).

Uso (desde cualquier carpeta):
    python 01_inferir.py --root "D:/.../Etiquetado-Musical"            # todo
    python 01_inferir.py --root "..." --limit 20                       # prueba rapida
Salida: <root>/resultados/{val,test}_{ids,y,pred_MTT_musicnn,pred_MTT_vgg}.npy
Es reanudable: si se corta, volve a correr el mismo comando y sigue donde quedo.
"""
import argparse
import ast
import os
import sys
import time

import numpy as np
import librosa
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

MODELOS = ["MTT_musicnn", "MTT_vgg"]
SPLITS = {"val": "val_gt_mtt.tsv", "test": "test_gt_mtt.tsv"}


def leer_split(root, nombre_gt):
    """Devuelve ids (str), rutas relativas a dataset/ y matriz de etiquetas (N, 50)."""
    idx = {}
    with open(os.path.join(root, "index_mtt.tsv"), encoding="utf8") as f:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) >= 2:
                idx[partes[0]] = partes[1]
    ids, rutas, Y = [], [], []
    with open(os.path.join(root, nombre_gt), encoding="utf8") as f:
        for linea in f:
            if not linea.strip():
                continue
            cid, vec = linea.rstrip("\n").split("\t")
            ids.append(cid)
            rutas.append(idx[cid])
            Y.append(ast.literal_eval(vec))
    return ids, rutas, np.array(Y, dtype=np.int8)


def mapa_carpetas(dataset_dir):
    """Letra de carpeta (0-9, a-f) -> ruta real. Funciona con dataset/c/... y con dataset/Validacion/c/..."""
    mapa = {}
    for dirpath, dirnames, _ in os.walk(dataset_dir):
        for d in dirnames:
            if len(d) == 1 and d in "0123456789abcdef":
                mapa[d] = os.path.join(dirpath, d)
    return mapa


def parches_logmel(ruta_audio, cfg, n_frames):
    """Mismo preprocesado que musicnn.extractor: mel -> log10(10000*mel+1) -> parches de 3 s sin solapamiento.
    (Se calcula en float32; el extractor original lo hace en float16.)"""
    audio, sr = librosa.load(ruta_audio, sr=cfg.SR)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=cfg.FFT_HOP,
                                         n_fft=cfg.FFT_SIZE, n_mels=cfg.N_MELS).T
    mel = np.log10(10000.0 * mel.astype(np.float32) + 1.0)
    ultimo = mel.shape[0] - n_frames + 1
    parches = [mel[t:t + n_frames] for t in range(0, ultimo, n_frames)]
    return np.stack(parches).astype(np.float32)  # (n_parches, 187, 96)


class Modelo:
    """Un modelo musicnn en su propio grafo/sesion, cargado una sola vez."""

    def __init__(self, nombre, musicnn_dir, models, cfg, n_frames):
        self.nombre = nombre
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("model"):
                self.x = tf.compat.v1.placeholder(tf.float32, [None, n_frames, cfg.N_MELS])
                self.is_training = tf.compat.v1.placeholder(tf.bool)
                salidas = models.define_model(self.x, self.is_training, nombre, len(cfg.MTT_LABELS))
                self.y = tf.nn.sigmoid(salidas[0])
            self.sess = tf.compat.v1.Session(graph=self.graph)
            self.sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            # Sin try/except: si falla la carga queremos enterarnos (el extractor original lo silencia).
            saver.restore(self.sess, os.path.join(musicnn_dir, nombre) + "/")

    def predecir(self, parches):
        p = self.sess.run(self.y, feed_dict={self.x: parches, self.is_training: False})
        return p.mean(axis=0)  # promedio sobre los parches del clip -> (50,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="carpeta Etiquetado-Musical")
    ap.add_argument("--limit", type=int, default=None, help="solo los primeros N clips por split (prueba)")
    ap.add_argument("--out", default=None, help="carpeta de salida (default: <root>/resultados)")
    ap.add_argument("--max-seg", type=float, default=None,
                    help="cortar prolijamente tras N segundos (se reanuda al volver a correr)")
    args = ap.parse_args()
    T_INICIO = time.time()

    root = os.path.abspath(args.root)
    out = args.out or os.path.join(root, "resultados")
    os.makedirs(out, exist_ok=True)

    musicnn_repo = os.path.join(root, "musicnn-master")
    sys.path.insert(0, musicnn_repo)
    from musicnn import models, configuration as cfg  # noqa: E402

    n_frames = librosa.time_to_frames(3, sr=cfg.SR, n_fft=cfg.FFT_SIZE, hop_length=cfg.FFT_HOP) + 1  # 187
    musicnn_dir = os.path.join(musicnn_repo, "musicnn")
    modelos = [Modelo(m, musicnn_dir, models, cfg, n_frames) for m in MODELOS]

    # chequeo: el orden de tags del TP (output_labels_mtt.txt) debe ser el del modelo
    with open(os.path.join(root, "output_labels_mtt.txt"), encoding="utf8") as f:
        txt = f.read()
    etiquetas_txt = ast.literal_eval(txt.split("=", 1)[1].strip())
    assert etiquetas_txt == cfg.MTT_LABELS, "El orden de etiquetas no coincide"

    carpetas = mapa_carpetas(os.path.join(root, "dataset"))
    for split, archivo_gt in SPLITS.items():
        ids, rutas, Y = leer_split(root, archivo_gt)
        faltan = {r.split("/")[0] for r in rutas} - set(carpetas)
        assert not faltan, f"No encuentro las carpetas {faltan} dentro de dataset/"
        if args.limit:
            ids, rutas, Y = ids[:args.limit], rutas[:args.limit], Y[:args.limit]
        N = len(ids)
        np.save(os.path.join(out, f"{split}_ids.npy"), np.array(ids))
        np.save(os.path.join(out, f"{split}_y.npy"), Y)
        np.save(os.path.join(out, f"{split}_rutas.npy"), np.array(rutas))

        parcial = os.path.join(out, f"{split}_parcial.npz")
        pred = {m: np.full((N, 50), np.nan, dtype=np.float32) for m in MODELOS}
        hecho = np.zeros(N, dtype=bool)
        if os.path.exists(parcial) and not args.limit:
            z = np.load(parcial)
            hecho = z["hecho"]
            for m in MODELOS:
                pred[m] = z[m]
            print(f"[{split}] reanudando: {hecho.sum()}/{N} ya hechos")

        t0 = time.time()
        procesados = 0
        interrumpido = False
        for i in range(N):
            if hecho[i]:
                continue
            if args.max_seg and time.time() - T_INICIO > args.max_seg:
                interrumpido = True
                break
            carpeta, archivo = rutas[i].split("/", 1)
            ruta = os.path.join(carpetas[carpeta], archivo)
            try:
                parches = parches_logmel(ruta, cfg, n_frames)
                for mod in modelos:
                    pred[mod.nombre][i] = mod.predecir(parches)
                hecho[i] = True
            except Exception as e:  # clip ilegible: queda en NaN y se reporta
                print(f"  ERROR en {rutas[i]}: {e}")
            procesados += 1
            if procesados % 100 == 0:
                vel = procesados / (time.time() - t0)
                print(f"[{split}] {hecho.sum()}/{N}  ({vel:.1f} clips/s)", flush=True)
            if procesados % 200 == 0 and not args.limit:
                np.savez(parcial, hecho=hecho, **pred)

        if not args.limit:
            np.savez(parcial, hecho=hecho, **pred)
        if interrumpido:
            print(f"[{split}] pausa por --max-seg: {hecho.sum()}/{N} hechos. Volve a correr el mismo comando para seguir.")
            return
        for m in MODELOS:
            np.save(os.path.join(out, f"{split}_pred_{m}.npy"), pred[m])
        print(f"[{split}] listo: {hecho.sum()}/{N} clips OK, {(~hecho).sum()} con error")
    print("TODO LISTO")


if __name__ == "__main__":
    main()
