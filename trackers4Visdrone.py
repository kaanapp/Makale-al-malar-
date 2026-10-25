import os
import tempfile
import numpy as np
import pandas as pd

# --- NUMPY 2.0+ UYUMLULUK YAMASI ---
if not hasattr(np, "asfarray"):
  np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
if not hasattr(np, "long"):
  np.long = np.int64
if not hasattr(np, "float"):
  np.float = float
if not hasattr(np, "int"):
  np.int = int

import motmetrics as mm

# Modeliniz sadece yaya/insan bulduğu için VisDrone sınıflarından sadece bunları alıyoruz:
# 1: Pedestrian (Yaya), 2: People (İnsan grubu)
VISDRONE_PEDESTRIAN_CLASSES = [1, 2]


def get_clean_model_name(filename):
  name = os.path.splitext(filename)[0]
  if "_" in name:
    return name.split("_", 1)[1]
  return name


def prepare_visdrone_files(gt_path, pred_path, video_name, model_name):
  # VisDrone anotasyonları yükleniyor
  gt_df = pd.read_csv(gt_path, header=None)

  # VisDrone Sınıf ve Skor Filtrelemesi:
  # 7. sütun (indeks 6: score/evaluation flag) > 0 ve
  # 8. sütun (indeks 7: object_category) Yaya sınıfları [1, 2] içinde olanlar
  if gt_df.shape[1] >= 8:
    gt_df = gt_df[
        (gt_df[6] > 0) & (gt_df[7].isin(VISDRONE_PEDESTRIAN_CLASSES))
    ]

  pred_df = pd.read_csv(pred_path, header=None)

  if gt_df.empty or pred_df.empty:
    return None, None

  gt_min, gt_max = int(gt_df[0].min()), int(gt_df[0].max())
  pred_min, pred_max = int(pred_df[0].min()), int(pred_df[0].max())

  # 0-index kayması varsa otomatik +1 ekle
  if pred_min == 0 and gt_min == 1:
    pred_df[0] = pred_df[0] + 1
    pred_min, pred_max = int(pred_df[0].min()), int(pred_df[0].max())
    print(f"  [DÜZELTME] {video_name} ({model_name}): Kareler +1 kaydırıldı.")

  # Model logunda eksik frame varsa GT'yi modelin kare aralığına hizala
  if 0 < (gt_max - pred_max) <= 2:
      gt_df = gt_df[gt_df[0] <= pred_max]

  # motmetrics formatı için 7. sütunu (confidence/flag) 1 yapıyoruz
  gt_df[6] = 1

  t_gt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
  gt_df.to_csv(t_gt.name, header=False, index=False)
  t_gt.close()

  t_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
  pred_df.to_csv(t_pred.name, header=False, index=False)
  t_pred.close()

  return t_gt.name, t_pred.name


def evaluate_visdrone(
    videos_root_dir="./trackers/visdrone_videolar",
    output_excel="VisDrone_Yaya_Sonuclar.xlsx",
):
  mh = mm.metrics.create()
  metrics = [
      "mota",
      "idf1",
      "motp",
      "idp",
      "idr",
      "mostly_tracked",
      "partially_tracked",
      "mostly_lost",
      "num_false_positives",
      "num_misses",
      "num_switches",
      "num_fragmentations",
      "num_objects",
      "num_unique_objects",
  ]

  if not os.path.exists(videos_root_dir):
    print(f"[HATA] '{videos_root_dir}' klasörü bulunamadı!")
    return

  video_folders = sorted([
      d
      for d in os.listdir(videos_root_dir)
      if os.path.isdir(os.path.join(videos_root_dir, d))
  ])

  model_to_videos = {}

  for vf in video_folders:
    v_path = os.path.join(videos_root_dir, vf)
    gt_path = os.path.join(v_path, "gt.txt")
    if not os.path.exists(gt_path):
      continue

    for f in sorted(os.listdir(v_path)):
      if f.endswith(".txt") and f.lower() != "gt.txt":
        model_name = get_clean_model_name(f)
        pred_path = os.path.join(v_path, f)
        if model_name not in model_to_videos:
          model_to_videos[model_name] = []
        model_to_videos[model_name].append((vf, pred_path, gt_path))

  overall_list = []
  excel_writer = pd.ExcelWriter(output_excel, engine="openpyxl")

  for model_name, records in sorted(model_to_videos.items()):
    print(f"\n==========================================")
    print(f"VisDrone Yaya Modeli Hesaplanıyor: {model_name}")
    print(f"==========================================")
    accs, seq_names, temp_files = [], [], []

    for vf, pred_path, gt_path in records:
      c_gt, c_pred = prepare_visdrone_files(gt_path, pred_path, vf, model_name)
      if c_gt is None:
        print(f"  [Uyarı] {vf} için geçerli yaya verisi bulunamadı.")
        continue

      temp_files.extend([c_gt, c_pred])

      gt = mm.io.loadtxt(c_gt, fmt="mot15-2D", min_confidence=1)
      ts = mm.io.loadtxt(c_pred, fmt="mot15-2D")

      acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
      accs.append(acc)
      seq_names.append(vf)

    if accs:
      summary = mh.compute_many(
          accs, metrics=metrics, names=seq_names, generate_overall=True
      )

      print(
          mm.io.render_summary(
              summary,
              formatters=mh.formatters,
              namemap=mm.io.motchallenge_metric_names,
          )
      )

      for col in ["mota", "idf1", "motp", "idp", "idr"]:
        if col in summary.columns:
          summary[col] = (summary[col] * 100).round(2)

      summary.rename(columns=mm.io.motchallenge_metric_names, inplace=True)
      summary.to_excel(excel_writer, sheet_name=model_name[:31])

      overall_row = summary.loc[["OVERALL"]].copy()
      overall_row.index = [model_name]
      overall_list.append(overall_row)

    for tf in temp_files:
      if os.path.exists(tf):
        os.remove(tf)

  if overall_list:
    final_df = pd.concat(overall_list)
    final_df.to_excel(excel_writer, sheet_name="Genel_Ortalama_Kiyaslama")
    print("\n" + "#" * 80)
    print("VİSDRONE YAYA TESPİTİ GENEL ORTALAMA (OVERALL) TABLOSU:")
    print("#" * 80)
    print(final_df.to_string())
    print(f"\n[BAŞARILI] Tablonuz '{output_excel}' dosyasına kaydedildi.")

  excel_writer.close()


if __name__ == "__main__":
  evaluate_visdrone("trackers/visdrone_videolar")