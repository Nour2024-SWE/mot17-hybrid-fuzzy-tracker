#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MOT17 HYBRID FUZZY TRACKING — IMPROVED VERSION v2.0
================================================================================
التحسينات المضافة (بدون تغيير بنية النماذج):
  1. معاملات كشف أكثر حساسية (YOLO_CONF=0.06, FRCNN_CONF=0.35, YOLO_IOU=0.50)
     → رفع Recall وتقليل False Negatives
  2. حدود مراقبة FIS أكثر مرونة (MERGE_BASE_THR=0.52, iou_gate=0.25)
     → التقاط كشوفات مشتركة أكثر
  3. معامل FIS Detector محسَّن (دوال عضوية أحدث حدوداً)
     → رفع ثقة الكشوفات الصحيحة بشكل أدق
  4. Sugeno defuzzification محسَّن (دعم متتبع أقوى للمسارات المستقرة)
     → تقليل Identity Switches
  5. TrackMemoryStabilizer بذاكرة أطول (max_age=100, min_hits=2)
     → إعادة ربط أفضل بعد الانسداد
  6. إضافة remove_duplicate_tracks() لمعالجة التشظي (Fragmentation)
  7. تحسين interpolate_tracks() بملء ثغرات أكبر للمسارات المتواصلة
  8. إعدادات خاصة لكل تسلسل مُحسَّنة بعناية
================================================================================
"""

import os, math, csv, zipfile, warnings
import numpy as np
import cv2
import torch
import torchvision
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

from ultralytics import YOLO

try:
    from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
    print("[OK] ByteTrack loaded")
except Exception as e:
    print(f"[WARN] ByteTrack import failed: {e}")
    ByteTrack = None

try:
    from boxmot.trackers.bbox.botsort.botsort import BotSort as BoTSORT
    print("[OK] BoTSORT (BotSort) loaded")
except Exception as e:
    print(f"[WARN] BoTSORT import failed: {e}")
    BoTSORT = None


# ==========================================================================================
# CONFIG
# ==========================================================================================

class Config:
    # ── Kaggle paths ──────────────────────────────────────────────────────────
    KAGGLE_MOT17     = Path("/kaggle/input/datasets/wenhoujinjust/mot-17/MOT17")
    ROOT_DIR         = KAGGLE_MOT17 / "train"
    OUTPUT_DIR       = Path("/kaggle/working/mot17_output")
    ZIP_PATH         = Path("/kaggle/working/MOT17_submission.zip")
    REID_WEIGHTS     = Path("/kaggle/working/osnet_x1_0_msmt17.pt")

    # ── FAST MODE ─────────────────────────────────────────────────────────────
    FAST_MODE        = True
    FAST_IMG_SIZE    = 640
    FAST_BATCH       = 16
    SKIP_FRCNN       = True
    MAX_FRAMES       = None

    # ── FINE-TUNE CONFIG ──────────────────────────────────────────────────────
    FINETUNE_ENABLED     = True    # ← True لتدريب YOLO على MOT17 قبل التتبع
    FINETUNE_EPOCHS      = 50      # عدد epochs (30 سريع، 50 أفضل، 100 الأفضل)
    FINETUNE_IMGSZ       = 1280    # حجم الصورة للتدريب (1280 أفضل دقة)
    FINETUNE_BATCH       = 4       # batch size (4 للـ GPU المحدود)
    FINETUNE_LR          = 0.001   # learning rate
    FINETUNE_PATIENCE    = 10      # early stopping
    FINETUNE_OUTPUT      = Path("/kaggle/working/yolo_finetune")
    FINETUNE_DATASET     = Path("/kaggle/working/mot17_yolo_dataset")
    FINETUNE_WEIGHTS     = Path("/kaggle/working/yolo_finetune/mot17/weights/best.pt")
    FINETUNE_BASE_MODEL  = "yolov8x.pt"   # النموذج الأساسي للتدريب
    FINETUNE_SKIP_IF_EXISTS = True  # ← True لتخطي التدريب إن كان الملف موجوداً
    # ─────────────────────────────────────────────────────────────────────────

    YOLO_WEIGHTS = "yolov8x.pt"
    YOLO_CONF    = 0.06   # أعمق recall مع FP مقبول
    YOLO_IOU     = 0.50   # أقل تداخل → كشف أكثر
    FRCNN_CONF   = 0.35   # Faster R-CNN threshold أقل صرامة
    TOPK_DETS    = 120    # سماح بكشوفات أكثر

    BYTETRACK_TRACK_THRESH = 0.18   # قبول مسارات بثقة أقل
    BYTETRACK_MATCH_THRESH = 0.78   # مطابقة أكثر دقة
    BYTETRACK_TRACK_BUFFER = 200    # ذاكرة أطول

    BOTSORT_TRACK_THRESH   = 0.18
    BOTSORT_MATCH_THRESH   = 0.78
    BOTSORT_TRACK_BUFFER   = 200

    MIN_TRACK_LENGTH      = 2
    MIN_TRACK_CONF        = 0.06   # قبول مسارات بثقة أدنى
    USE_INTERPOLATION     = True
    INTERPOLATION_MAX_GAP = 55     # ملء ثغرات أكبر

    DEVICE       = 0 if torch.cuda.is_available() else "cpu"
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16         = torch.cuda.is_available()

    SEQUENCE_SPECIFIC = {
        "MOT17-01": {"YOLO_CONF": 0.10,  "TRACK_BUFFER": 240, "MATCH_THRESH": 0.80,
                     "INTERPOLATION_MAX_GAP": 55, "MIN_TRACK_LENGTH": 2,
                     "RECOVERY_IOU": 0.58, "KEEP_IOU": 0.56},
        "MOT17-03": {"YOLO_CONF": 0.010, "TRACK_BUFFER": 270, "MATCH_THRESH": 0.65,
                     "INTERPOLATION_MAX_GAP": 70, "RECOVERY_IOU": 0.60, "KEEP_IOU": 0.58,
                     "SWITCH_COOLDOWN": 12, "SWITCH_PENALTY": 0.82,
                     "STABLE_MERGE_MARGIN": 0.04, "MERGE_BASE_THR": 0.55,
                     "UNMATCHED_BOT_CONF_MIN": 0.18},
        "MOT17-06": {"YOLO_CONF": 0.012, "TRACK_BUFFER": 260, "MATCH_THRESH": 0.62,
                     "INTERPOLATION_MAX_GAP": 65, "RECOVERY_IOU": 0.60, "KEEP_IOU": 0.58,
                     "SWITCH_COOLDOWN": 14, "SWITCH_PENALTY": 0.82,
                     "STABLE_MERGE_MARGIN": 0.05, "MERGE_BASE_THR": 0.55,
                     "UNMATCHED_BOT_CONF_MIN": 0.16},
        "MOT17-07": {"YOLO_CONF": 0.10,  "TRACK_BUFFER": 240, "MATCH_THRESH": 0.70,
                     "INTERPOLATION_MAX_GAP": 55, "RECOVERY_IOU": 0.58, "KEEP_IOU": 0.56},
        "MOT17-08": {"YOLO_CONF": 0.14, "FRCNN_CONF": 0.38, "TRACK_THRESH": 0.20,
                     "TRACK_BUFFER": 180, "MATCH_THRESH": 0.70,
                     "INTERPOLATION_MAX_GAP": 40, "MIN_TRACK_CONF": 0.08,
                     "UNMATCHED_BOT_CONF_MIN": 0.22, "BOT_OVERLAP_GATE": 0.33,
                     "MEMORY_MIN_HITS": 3, "MERGE_BASE_THR": 0.58,
                     "STRICT_MOTA_MODE": True, "TOPK_DETS": 90, "MEMORY_MAX_AGE": 45,
                     "ADAPTIVE_MATCH_MIN_THR": 0.58, "ADAPTIVE_MATCH_MAX_THR": 0.78,
                     "PRE_FILTER_MIN_CONF": 0.30, "FIS_CONF_GATE": 0.30,
                     "BLOCK_FRCNN_ONLY": False, "YOLO_ONLY_MIN_CONF": 0.45,
                     "SINGLE_DETECTOR_PENALTY": 0.70, "TEMPORAL_CONFIRM_IOU": 0.55,
                     "TEMPORAL_BIRTH_MIN_CONF": 0.60},
        "MOT17-12": {"YOLO_CONF": 0.14, "FRCNN_CONF": 0.38, "TRACK_THRESH": 0.20,
                     "TRACK_BUFFER": 175, "MATCH_THRESH": 0.73,
                     "MIN_TRACK_LENGTH": 2, "INTERPOLATION_MAX_GAP": 35,
                     "MIN_TRACK_CONF": 0.08, "UNMATCHED_BOT_CONF_MIN": 0.24,
                     "BOT_OVERLAP_GATE": 0.37, "MEMORY_MIN_HITS": 4,
                     "MERGE_BASE_THR": 0.58, "STRICT_MOTA_MODE": True,
                     "TOPK_DETS": 80, "MEMORY_MAX_AGE": 40,
                     "ADAPTIVE_MATCH_MIN_THR": 0.58, "ADAPTIVE_MATCH_MAX_THR": 0.78,
                     "PRE_FILTER_MIN_CONF": 0.32, "FIS_CONF_GATE": 0.40,
                     "BLOCK_FRCNN_ONLY": False, "YOLO_ONLY_MIN_CONF": 0.50,
                     "SINGLE_DETECTOR_PENALTY": 0.60, "TEMPORAL_CONFIRM_IOU": 0.42,
                     "TEMPORAL_BIRTH_MIN_CONF": 0.58},
        "MOT17-14": {"YOLO_CONF": 0.012, "TRACK_BUFFER": 290, "MATCH_THRESH": 0.55,
                     "INTERPOLATION_MAX_GAP": 75, "RECOVERY_IOU": 0.60, "KEEP_IOU": 0.58,
                     "SWITCH_COOLDOWN": 14, "SWITCH_PENALTY": 0.82,
                     "STABLE_MERGE_MARGIN": 0.05, "UNMATCHED_BOT_CONF_MIN": 0.18,
                     "MERGE_BASE_THR": 0.55},
    }

    @staticmethod
    def get_seq_config(seq_name):
        base = '-'.join(seq_name.split('-')[:2])
        spec = dict(Config.SEQUENCE_SPECIFIC.get(base, {}))
        if "YOLO_CONF" in spec:
            spec["YOLO_CONF"] = float(np.clip(spec["YOLO_CONF"], 0.008, 0.25))
        if "FRCNN_CONF" in spec:
            spec["FRCNN_CONF"] = float(np.clip(spec["FRCNN_CONF"], 0.30, 0.80))
        if "TRACK_THRESH" in spec:
            spec["TRACK_THRESH"] = float(np.clip(spec["TRACK_THRESH"], 0.15, 0.45))
        if "TRACK_BUFFER" in spec:
            min_buffer = 120 if spec.get("STRICT_MOTA_MODE") else 180
            spec["TRACK_BUFFER"] = int(max(min_buffer, spec["TRACK_BUFFER"]))
        if "INTERPOLATION_MAX_GAP" in spec:
            min_gap = 25 if spec.get("STRICT_MOTA_MODE") else 45
            spec["INTERPOLATION_MAX_GAP"] = int(max(min_gap, spec["INTERPOLATION_MAX_GAP"]))
        if "MIN_TRACK_CONF" in spec:
            spec["MIN_TRACK_CONF"] = float(np.clip(spec["MIN_TRACK_CONF"], 0.06, 0.18))
        if "UNMATCHED_BOT_CONF_MIN" in spec:
            spec["UNMATCHED_BOT_CONF_MIN"] = float(np.clip(spec["UNMATCHED_BOT_CONF_MIN"], 0.14, 0.30))
        if "RECOVERY_IOU" in spec:
            spec["RECOVERY_IOU"] = float(np.clip(spec["RECOVERY_IOU"], 0.55, 0.68))
        if "KEEP_IOU" in spec:
            spec["KEEP_IOU"] = float(np.clip(spec["KEEP_IOU"], 0.52, 0.66))
        if "SWITCH_COOLDOWN" in spec:
            spec["SWITCH_COOLDOWN"] = int(np.clip(spec["SWITCH_COOLDOWN"], 8, 20))
        if "SWITCH_PENALTY" in spec:
            spec["SWITCH_PENALTY"] = float(np.clip(spec["SWITCH_PENALTY"], 0.78, 0.95))
        if "STABLE_MERGE_MARGIN" in spec:
            spec["STABLE_MERGE_MARGIN"] = float(np.clip(spec["STABLE_MERGE_MARGIN"], 0.0, 0.06))
        if "BOT_OVERLAP_GATE" in spec:
            spec["BOT_OVERLAP_GATE"] = float(np.clip(spec["BOT_OVERLAP_GATE"], 0.22, 0.45))
        if "MERGE_BASE_THR" in spec:
            spec["MERGE_BASE_THR"] = float(np.clip(spec["MERGE_BASE_THR"], 0.52, 0.62))
        if "MEMORY_MIN_HITS" in spec:
            spec["MEMORY_MIN_HITS"] = int(np.clip(spec["MEMORY_MIN_HITS"], 2, 5))
        if "TOPK_DETS" in spec:
            spec["TOPK_DETS"] = int(np.clip(spec["TOPK_DETS"], 20, 130))
        if "PRE_FILTER_MIN_CONF" in spec:
            spec["PRE_FILTER_MIN_CONF"] = float(np.clip(spec["PRE_FILTER_MIN_CONF"], 0.28, 0.55))
        if "FIS_CONF_GATE" in spec:
            spec["FIS_CONF_GATE"] = float(np.clip(spec["FIS_CONF_GATE"], 0.28, 0.60))
        if "YOLO_ONLY_MIN_CONF" in spec:
            spec["YOLO_ONLY_MIN_CONF"] = float(np.clip(spec["YOLO_ONLY_MIN_CONF"], 0.40, 0.90))
        if "SINGLE_DETECTOR_PENALTY" in spec:
            spec["SINGLE_DETECTOR_PENALTY"] = float(np.clip(spec["SINGLE_DETECTOR_PENALTY"], 0.45, 0.80))
        if "TEMPORAL_CONFIRM_IOU" in spec:
            spec["TEMPORAL_CONFIRM_IOU"] = float(np.clip(spec["TEMPORAL_CONFIRM_IOU"], 0.38, 0.60))
        if "TEMPORAL_BIRTH_MIN_CONF" in spec:
            spec["TEMPORAL_BIRTH_MIN_CONF"] = float(np.clip(spec["TEMPORAL_BIRTH_MIN_CONF"], 0.55, 0.85))
        if "MEMORY_MAX_AGE" in spec:
            spec["MEMORY_MAX_AGE"] = int(np.clip(spec["MEMORY_MAX_AGE"], 25, 140))
        if "ADAPTIVE_MATCH_MIN_THR" in spec:
            spec["ADAPTIVE_MATCH_MIN_THR"] = float(np.clip(spec["ADAPTIVE_MATCH_MIN_THR"], 0.55, 0.70))
        if "ADAPTIVE_MATCH_MAX_THR" in spec:
            spec["ADAPTIVE_MATCH_MAX_THR"] = float(np.clip(spec["ADAPTIVE_MATCH_MAX_THR"], 0.70, 0.85))
        return spec

config = Config()

MOT17_TEST_BASE_IDS  = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14"]
MOT17_TRAIN_BASE_IDS = ["02","04","05","09","10","11","13"]   # التسلسلات التي لها GT في train
MOT17_DETECTORS      = ["DPM","FRCNN","SDP"]
MOT17_ALL_42         = [f"MOT17-{sid}-{det}" for sid in MOT17_TEST_BASE_IDS for det in MOT17_DETECTORS]

# ── وضع التشغيل: train = تقييم محلي | test = إنتاج submission ──
RUN_MODE = "train"   # غيّر إلى "test" عند الرفع الرسمي


# ==========================================================================================
# HELPERS
# ==========================================================================================

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def compute_iou(box1, box2):
    xA = max(float(box1[0]), float(box2[0])); yA = max(float(box1[1]), float(box2[1]))
    xB = min(float(box1[2]), float(box2[2])); yB = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, xB-xA) * max(0.0, yB-yA)
    a1 = max(0.0, float(box1[2]-box1[0])) * max(0.0, float(box1[3]-box1[1]))
    a2 = max(0.0, float(box2[2]-box2[0])) * max(0.0, float(box2[3]-box2[1]))
    return inter / (a1 + a2 - inter + 1e-9)

def compute_motion_similarity(box1, box2):
    c1x=(box1[0]+box1[2])/2.0; c1y=(box1[1]+box1[3])/2.0
    c2x=(box2[0]+box2[2])/2.0; c2y=(box2[1]+box2[3])/2.0
    dist = math.sqrt((c1x-c2x)**2+(c1y-c2y)**2)
    diag = math.sqrt((box1[2]-box1[0])**2+(box1[3]-box1[1])**2)+1e-9
    return float(max(0.0, 1.0-dist/(diag*2.0)))

def compute_velocity_direction_similarity(p1, c1, p2, c2):
    if p1 is None or p2 is None: return 0.5
    def vel(p,c): return np.array([(c[0]+c[2])/2-(p[0]+p[2])/2,(c[1]+c[3])/2-(p[1]+p[3])/2],dtype=np.float64)
    v1=vel(p1,c1); v2=vel(p2,c2)
    cos=np.dot(v1,v2)/(np.linalg.norm(v1)+1e-9)/(np.linalg.norm(v2)+1e-9)
    return float(np.clip((cos+1.0)/2.0,0.0,1.0))

def normalized_center_proximity(A, B):
    cx1=(A[0]+A[2])/2.0; cy1=(A[1]+A[3])/2.0
    cx2=(B[0]+B[2])/2.0; cy2=(B[1]+B[3])/2.0
    dist=math.sqrt((cx1-cx2)**2+(cy1-cy2)**2)
    dA=math.sqrt((A[2]-A[0])**2+(A[3]-A[1])**2)
    dB=math.sqrt((B[2]-B[0])**2+(B[3]-B[1])**2)
    return float(np.clip(1.0-min(1.0,dist/(max(dA,dB)+1e-9)),0.0,1.0))

def compute_history_consistency(curr, prev):
    if prev is None: return 0.5
    return 0.5*compute_iou(curr,prev)+0.5*normalized_center_proximity(curr,prev)

def compute_adaptive_merge_threshold(density, base_thr=0.52):
    return float(np.clip(base_thr+0.18*min(1.0,density/80.0),0.52,0.72))

def compute_adaptive_match_threshold(proximity_score, min_thr=0.55, max_thr=0.75):
    adaptive_thr = max_thr - (max_thr - min_thr) * float(np.clip(proximity_score, 0.0, 1.0))
    return float(np.clip(adaptive_thr, min_thr, max_thr))

def is_valid_number(x):
    try: return np.isfinite(float(x))
    except: return False

def pre_filter_fused_detections(dets, topk_dets=60, min_conf_override=None):
    if len(dets) == 0:
        return dets

    density = len(dets)
    if density > 50:
        min_conf = 0.45 - 0.02 * min(1.0, (density - 50) / 30)
    elif density > 35:
        min_conf = 0.42 - 0.015 * (density - 35) / 15
    elif density > 20:
        min_conf = 0.36 - 0.02 * (density - 20) / 15
    else:
        min_conf = 0.30 - 0.01 * density / 20

    min_conf = max(0.18, min(0.45, min_conf))
    if min_conf_override is not None:
        min_conf = max(min_conf, float(min_conf_override))
    min_w, min_h = 10, 20  # accept smaller boxes → fewer FN

    clean = []
    for d in dets:
        x1, y1, x2, y2, conf, _ = d
        if conf < min_conf:
            continue
        if (x2 - x1) < min_w or (y2 - y1) < min_h:
            continue
        clean.append(d)

    if clean:
        clean = sorted(clean, key=lambda x: x[4], reverse=True)
        if len(clean) > topk_dets:
            clean = clean[:topk_dets]

    return np.array(clean, dtype=np.float32) if clean else np.zeros((0, 6), dtype=np.float32)

def temporally_confirm_detections(
    dets, prev_dets, memory_stabilizer=None, iou_thr=0.45, existing_iou=0.32, high_conf_pass=0.65
):
    if len(dets) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    prev_arr = prev_dets if len(prev_dets) > 0 else np.zeros((0, 6), dtype=np.float32)
    has_recent_memory = False
    if memory_stabilizer is not None:
        has_recent_memory = any((memory_stabilizer.frame_idx - entry["last_seen"]) <= 1 for entry in memory_stabilizer.memory.values())

    # First-frame / cold-start behavior: allow only strong detections through.
    if len(prev_arr) == 0 and not has_recent_memory:
        keep = dets[:, 4] >= float(high_conf_pass)
        return dets[keep] if np.any(keep) else np.zeros((0, 6), dtype=np.float32)

    confirmed = []

    for det in dets:
        box = det[:4]
        conf = float(det[4])
        supported = False

        # Existing tracks should not be blocked by birth control.
        if memory_stabilizer is not None:
            for entry in memory_stabilizer.memory.values():
                if (memory_stabilizer.frame_idx - entry["last_seen"]) > 1:
                    continue
                if compute_iou(box, entry["box"]) >= existing_iou:
                    supported = True
                    break

        if supported:
            confirmed.append(det)
            continue

        if len(prev_arr) == 0 and not has_recent_memory and conf >= high_conf_pass:
            confirmed.append(det)
            continue

        if not supported:
            for prev in prev_arr:
                if compute_iou(box, prev[:4]) >= iou_thr:
                    supported = True
                    break

        if supported:
            confirmed.append(det)

    return np.array(confirmed, dtype=np.float32) if confirmed else np.zeros((0, 6), dtype=np.float32)

def has_recent_track_support(box, memory_stabilizer, existing_iou=0.28):
    if memory_stabilizer is None:
        return False
    for entry in memory_stabilizer.memory.values():
        if (memory_stabilizer.frame_idx - entry["last_seen"]) > 1:
            continue
        if compute_iou(box, entry["box"]) >= existing_iou:
            return True
    return False


# ==========================================================================================
# DETECTOR FIS — دمج ضبابي حقيقي للكواشف
# ==========================================================================================

class DetectorFusionFIS:
    """
    نظام استدلال ضبابي لدمج كواشف YOLO و Faster R-CNN
    
    المدخلات:
      - iou_score:    درجة التطابق المكاني بين كاشفَين (0-1)
      - conf_avg:     متوسط ثقة الكاشفَين (0-1)
    
    المخرج:
      - fusion_conf:  درجة ثقة الكشف المدموج (0-1)
    
    دوال العضوية: triangular (trimf)
    Defuzzification: centroid
    """
    def __init__(self):
        self.iou_score  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'iou_score')
        self.conf_avg   = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'conf_avg')
        self.fusion_conf= ctrl.Consequent(np.arange(0, 1.01, 0.01), 'fusion_conf')

        # دوال عضوية للمدخلات
        self.iou_score['low']    = fuzz.trimf(self.iou_score.universe,   [0.0, 0.0, 0.35])
        self.iou_score['medium'] = fuzz.trimf(self.iou_score.universe,   [0.15, 0.45, 0.75])
        self.iou_score['high']   = fuzz.trimf(self.iou_score.universe,   [0.55, 1.0, 1.0])

        self.conf_avg['low']     = fuzz.trimf(self.conf_avg.universe,    [0.0, 0.0, 0.35])
        self.conf_avg['medium']  = fuzz.trimf(self.conf_avg.universe,    [0.15, 0.45, 0.75])
        self.conf_avg['high']    = fuzz.trimf(self.conf_avg.universe,    [0.55, 1.0, 1.0])

        # دوال عضوية للمخرج (أرفع قيم للنتائج المشتركة)
        self.fusion_conf['very_low']  = fuzz.trimf(self.fusion_conf.universe, [0.0,  0.0,  0.20])
        self.fusion_conf['low']       = fuzz.trimf(self.fusion_conf.universe, [0.1,  0.28, 0.46])
        self.fusion_conf['medium']    = fuzz.trimf(self.fusion_conf.universe, [0.35, 0.55, 0.72])
        self.fusion_conf['high']      = fuzz.trimf(self.fusion_conf.universe, [0.62, 0.82, 0.96])
        self.fusion_conf['very_high'] = fuzz.trimf(self.fusion_conf.universe, [0.82, 1.0,  1.0])

        # قواعد IF-THEN الضبابية
        rules = [
            # كلا الكاشفَين يتفقان بثقة عالية → ثقة عالية جداً
            ctrl.Rule(self.iou_score['high']   & self.conf_avg['high'],   self.fusion_conf['very_high']),
            # تطابق جيد مع ثقة متوسطة → ثقة عالية
            ctrl.Rule(self.iou_score['high']   & self.conf_avg['medium'], self.fusion_conf['high']),
            # تطابق متوسط مع ثقة عالية → ثقة عالية
            ctrl.Rule(self.iou_score['medium'] & self.conf_avg['high'],   self.fusion_conf['high']),
            # تطابق متوسط مع ثقة متوسطة → ثقة متوسطة-عالية
            ctrl.Rule(self.iou_score['medium'] & self.conf_avg['medium'], self.fusion_conf['medium']),
            # تطابق ضعيف مع ثقة عالية → ثقة متوسطة
            ctrl.Rule(self.iou_score['low']    & self.conf_avg['high'],   self.fusion_conf['medium']),
            # تطابق ضعيف مع ثقة متوسطة → ثقة منخفضة
            ctrl.Rule(self.iou_score['low']    & self.conf_avg['medium'], self.fusion_conf['low']),
            # تطابق ضعيف مع ثقة منخفضة → ثقة منخفضة جداً
            ctrl.Rule(self.iou_score['low']    & self.conf_avg['low'],    self.fusion_conf['very_low']),
        ]

        self.system = ctrl.ControlSystem(rules)
        self.sim    = ctrl.ControlSystemSimulation(self.system)

    def compute(self, iou_score, conf_avg):
        try:
            self.sim.input['iou_score'] = float(np.clip(iou_score, 0, 1))
            self.sim.input['conf_avg']  = float(np.clip(conf_avg,  0, 1))
            self.sim.compute()
            return float(np.clip(self.sim.output['fusion_conf'], 0.0, 1.0))
        except Exception:
            self.sim = ctrl.ControlSystemSimulation(self.system)
            return float(np.clip(0.6*iou_score + 0.4*conf_avg, 0.0, 1.0))


# ==========================================================================================
# TRACKER FIS — دمج ضبابي حقيقي للمتتبعات (من mot17_balanced)
# ==========================================================================================

class TrackerProximityFIS:
    """FIS للقرب المكاني بين مسارَي ByteTrack وBoT-SORT"""
    def __init__(self):
        self.iou        = ctrl.Antecedent(np.arange(0,1.01,0.01), 'iou')
        self.center_sim = ctrl.Antecedent(np.arange(0,1.01,0.01), 'center_sim')
        self.size_ratio = ctrl.Antecedent(np.arange(0,1.01,0.01), 'size_ratio')
        self.proximity  = ctrl.Consequent(np.arange(0,1.01,0.01), 'proximity')

        for var, pts in [
            (self.iou,        ([0.0,0.0,0.35],[0.2,0.45,0.7],[0.55,1.0,1.0])),
            (self.center_sim, ([0.0,0.0,0.35],[0.2,0.5,0.8], [0.6,1.0,1.0])),
            (self.size_ratio, ([0.0,0.0,0.4], [0.25,0.5,0.75],[0.6,1.0,1.0])),
        ]:
            var['low']=fuzz.trimf(var.universe,pts[0]); var['medium']=fuzz.trimf(var.universe,pts[1]); var['high']=fuzz.trimf(var.universe,pts[2])

        self.proximity['very_low'] =fuzz.trimf(self.proximity.universe,[0.0, 0.0, 0.2])
        self.proximity['low']      =fuzz.trimf(self.proximity.universe,[0.1, 0.25,0.45])
        self.proximity['medium']   =fuzz.trimf(self.proximity.universe,[0.35,0.5, 0.65])
        self.proximity['high']     =fuzz.trimf(self.proximity.universe,[0.55,0.75,0.9])
        self.proximity['very_high']=fuzz.trimf(self.proximity.universe,[0.8, 1.0, 1.0])

        rules=[
            ctrl.Rule(self.iou['high']  &self.center_sim['high']  &self.size_ratio['high'],  self.proximity['very_high']),
            ctrl.Rule(self.iou['high']  &self.center_sim['high'],                             self.proximity['high']),
            ctrl.Rule(self.iou['high']  &self.size_ratio['medium'],                           self.proximity['high']),
            ctrl.Rule(self.iou['medium']&self.center_sim['high'],                             self.proximity['high']),
            ctrl.Rule(self.iou['medium']&self.center_sim['medium']&self.size_ratio['medium'], self.proximity['medium']),
            ctrl.Rule(self.iou['medium']&self.size_ratio['low'],                              self.proximity['medium']),
            ctrl.Rule(self.iou['low']   &self.center_sim['high'],                             self.proximity['medium']),
            ctrl.Rule(self.iou['low']   &self.center_sim['medium'],                           self.proximity['low']),
            ctrl.Rule(self.iou['low']   |self.center_sim['low'],                              self.proximity['very_low']),
        ]
        self.system=ctrl.ControlSystem(rules); self.sim=ctrl.ControlSystemSimulation(self.system)

    def compute(self,iou,center_sim,size_ratio):
        try:
            self.sim.input['iou']=float(np.clip(iou,0,1)); self.sim.input['center_sim']=float(np.clip(center_sim,0,1)); self.sim.input['size_ratio']=float(np.clip(size_ratio,0,1))
            self.sim.compute(); return float(np.clip(self.sim.output['proximity'],0.0,1.0))
        except:
            self.sim=ctrl.ControlSystemSimulation(self.system); return float(np.clip(0.5*iou+0.3*center_sim+0.2*size_ratio,0.0,1.0))


class TrackerMotionFIS:
    """FIS للحركة والاتساق الزمني بين المتتبعَين"""
    def __init__(self):
        self.motion_sim  =ctrl.Antecedent(np.arange(0,1.01,0.01),'motion_sim')
        self.vel_dir     =ctrl.Antecedent(np.arange(0,1.01,0.01),'vel_dir')
        self.conf_consist=ctrl.Antecedent(np.arange(0,1.01,0.01),'conf_consist')
        self.motion_score=ctrl.Consequent(np.arange(0,1.01,0.01),'motion_score')

        for var,pts in [
            (self.motion_sim,  ([0.0,0.0,0.35],[0.2,0.5,0.75],[0.6,1.0,1.0])),
            (self.vel_dir,     ([0.0,0.0,0.35],[0.2,0.5,0.8], [0.6,1.0,1.0])),
            (self.conf_consist,([0.0,0.0,0.4], [0.25,0.5,0.75],[0.6,1.0,1.0])),
        ]:
            var['low']=fuzz.trimf(var.universe,pts[0]); var['medium']=fuzz.trimf(var.universe,pts[1]); var['high']=fuzz.trimf(var.universe,pts[2])

        self.motion_score['very_low'] =fuzz.trimf(self.motion_score.universe,[0.0, 0.0, 0.2])
        self.motion_score['low']      =fuzz.trimf(self.motion_score.universe,[0.1, 0.25,0.45])
        self.motion_score['medium']   =fuzz.trimf(self.motion_score.universe,[0.35,0.5, 0.65])
        self.motion_score['high']     =fuzz.trimf(self.motion_score.universe,[0.55,0.75,0.9])
        self.motion_score['very_high']=fuzz.trimf(self.motion_score.universe,[0.8, 1.0, 1.0])

        rules=[
            ctrl.Rule(self.motion_sim['high']  &self.vel_dir['high']  &self.conf_consist['high'],  self.motion_score['very_high']),
            ctrl.Rule(self.motion_sim['high']  &self.vel_dir['high'],                              self.motion_score['high']),
            ctrl.Rule(self.motion_sim['high']  &self.conf_consist['high'],                         self.motion_score['high']),
            ctrl.Rule(self.motion_sim['medium']&self.vel_dir['high'],                              self.motion_score['high']),
            ctrl.Rule(self.motion_sim['medium']&self.vel_dir['medium']&self.conf_consist['medium'],self.motion_score['medium']),
            ctrl.Rule(self.motion_sim['medium']&self.conf_consist['low'],                          self.motion_score['medium']),
            ctrl.Rule(self.motion_sim['low']   &self.vel_dir['high'],                              self.motion_score['medium']),
            ctrl.Rule(self.motion_sim['low']   &self.vel_dir['medium'],                            self.motion_score['low']),
            ctrl.Rule(self.motion_sim['low']   |self.vel_dir['low'],                               self.motion_score['very_low']),
        ]
        self.system=ctrl.ControlSystem(rules); self.sim=ctrl.ControlSystemSimulation(self.system)

    def compute(self,motion_sim,vel_dir,conf_consist):
        try:
            self.sim.input['motion_sim']=float(np.clip(motion_sim,0,1)); self.sim.input['vel_dir']=float(np.clip(vel_dir,0,1)); self.sim.input['conf_consist']=float(np.clip(conf_consist,0,1))
            self.sim.compute(); return float(np.clip(self.sim.output['motion_score'],0.0,1.0))
        except:
            self.sim=ctrl.ControlSystemSimulation(self.system); return float(np.clip(0.5*motion_sim+0.3*vel_dir+0.2*conf_consist,0.0,1.0))


def fuzzy_tracker_decision(proximity_score, motion_score, history_score=0.5):
    """Sugeno-style defuzzification لقرار الدمج — معامِلات محسَّنة لتقليل IDSW"""
    def high(x): return max(0.0,min(1.0,(x-0.45)/0.55))
    def low(x):  return max(0.0,min(1.0,(0.55-x)/0.55))
    pe = 0.70*proximity_score + 0.30*history_score
    me = 0.60*motion_score    + 0.40*history_score
    r1=min(high(pe),high(me)); r2=min(high(pe),low(me))
    r3=min(low(pe), high(me)); r4=min(low(pe), low(me))
    # رفع r1 لتشجيع قبول التطابق الجيد
    num=(r1*0.96+r2*0.74+r3*0.70+r4*0.16)
    den=(r1+r2+r3+r4+1e-9)
    return float(np.clip(num/den,0.0,1.0))


class FuzzyTrackerFusionEngine:
    def __init__(self):
        print("[FIS] Initializing DetectorFusionFIS...")
        self.detector_fis = DetectorFusionFIS()
        print("[FIS] Initializing TrackerProximityFIS...")
        self.proximity_fis= TrackerProximityFIS()
        print("[FIS] Initializing TrackerMotionFIS...")
        self.motion_fis   = TrackerMotionFIS()
        print("[FIS] All FIS engines ready.")

    def compute_tracker(self,iou,center_sim,size_ratio,motion_sim,vel_dir,conf_consist,history):
        prox  = self.proximity_fis.compute(iou,center_sim,size_ratio)
        mot   = self.motion_fis.compute(motion_sim,vel_dir,conf_consist)
        final = fuzzy_tracker_decision(prox,mot,history)
        return {"proximity_score":float(np.clip(prox,0,1)),
                "motion_score":   float(np.clip(mot,0,1)),
                "final_score":    float(np.clip(final,0,1))}

TRACKER_FIS_ENGINE = FuzzyTrackerFusionEngine()


# ==========================================================================================
# TRACK MEMORY STABILIZER
# ==========================================================================================

class TrackMemoryStabilizer:
    def __init__(self, max_age=100, recovery_iou=0.58, min_hits=2, switch_cooldown=10):
        self.memory={}; self.max_age=max_age; self.recovery_iou=float(recovery_iou)
        self.min_hits=int(min_hits); self.switch_cooldown=int(switch_cooldown); self.frame_idx=0

    def step(self):
        self.frame_idx+=1
        dead=[t for t,v in self.memory.items() if self.frame_idx-v["last_seen"]>self.max_age]
        for t in dead: del self.memory[t]

    def update_track(self,tid,box,conf):
        tid=int(tid); prev=self.memory.get(tid)
        gap=999 if prev is None else self.frame_idx-prev["last_seen"]
        hits=1 if prev is None else prev.get("hits",0)+1
        streak=1 if prev is None or gap>1 else prev.get("streak",1)+1
        self.memory[tid]={"box":np.array(box[:4],dtype=np.float32),"conf":float(conf),
            "last_seen":self.frame_idx,"hits":hits,"streak":streak,
            "last_switch":(-10**9 if prev is None else prev.get("last_switch",-10**9))}

    def get_best_previous_match(self,box,min_iou=None,exclude_tid=None,only_lost=False):
        min_iou=self.recovery_iou if min_iou is None else float(min_iou)
        best_tid,best_iou,best_score=None,0.0,-1e9
        for tid,v in self.memory.items():
            if exclude_tid is not None and int(tid)==int(exclude_tid): continue
            age=self.frame_idx-v["last_seen"]
            if only_lost and age<=1: continue
            if v.get("hits",0)<self.min_hits: continue
            iou=compute_iou(v["box"],box[:4])
            if iou<min_iou: continue
            score=iou+0.04*min(v.get("streak",1),8)-0.008*max(0,age-1)  # reward streak, penalize age less
            if score>best_score: best_tid,best_iou,best_score=tid,iou,score
        return best_tid, best_iou

    def should_keep_current_id(self,tid,box,continuity_iou=0.54):
        v=self.memory.get(int(tid))
        if v is None or (self.frame_idx-v["last_seen"])>1: return False
        return compute_iou(v["box"],box[:4])>=continuity_iou

    def can_reassign(self,current_tid):
        v=self.memory.get(int(current_tid))
        if v is None: return True
        if (self.frame_idx-v.get("last_switch",-10**9))<self.switch_cooldown: return False
        if v.get("streak",0)>=self.min_hits and (self.frame_idx-v["last_seen"])<=1: return False
        return True

    def note_reassignment(self,from_tid,to_tid):
        if from_tid is None or int(from_tid)==int(to_tid): return
        for tid in [from_tid,to_tid]:
            v=self.memory.get(int(tid))
            if v is not None: v["last_switch"]=self.frame_idx

    def was_matched_last_frame(self,tid):
        v=self.memory.get(int(tid)); return False if v is None else (self.frame_idx-v["last_seen"])<=1

    def get_previous_box(self,tid):
        v=self.memory.get(int(tid)); return v["box"] if v else None


class TrajectorySmoother:
    def __init__(self,max_hist=5):
        self.hist=defaultdict(lambda:deque(maxlen=max_hist))
    def smooth(self,tid,box,speed=None):
        self.hist[int(tid)].append(np.array(box[:4],dtype=np.float32))
        arr=(np.stack(list(self.hist[int(tid)])[-2:],axis=0) if speed and speed>30.0
             else np.stack(self.hist[int(tid)],axis=0))
        return np.mean(arr,axis=0)


# ==========================================================================================
# CAMERA MOTION COMPENSATION
# ==========================================================================================

class CameraMotionCompensator:
    def __init__(self):
        self.prev_gray=None; self.prev_warp=np.eye(2,3,dtype=np.float32)

    def estimate(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray=gray; return np.eye(2,3,dtype=np.float32)
        warp=np.eye(2,3,dtype=np.float32)
        crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,1e-4)
        try:
            _,warp=cv2.findTransformECC(self.prev_gray,gray,warp,cv2.MOTION_EUCLIDEAN,crit,None,1)
            self.prev_warp=warp.copy()
        except:
            warp=self.prev_warp.copy()
        self.prev_gray=gray; return warp

    @staticmethod
    def compensate_box(box,warp):
        pts=np.array([[box[0],box[1],1.0],[box[2],box[3],1.0]],dtype=np.float32).T
        t=np.dot(warp,pts).T
        return np.array([t[0][0],t[0][1],t[1][0],t[1][1]],dtype=np.float32)


# ==========================================================================================
# DETECTORS: YOLO + Faster R-CNN (GPU)
# ==========================================================================================


# ==========================================================================================
# YOLO FINE-TUNE PIPELINE
# ==========================================================================================

def prepare_mot17_yolo_dataset(mot17_train_path, output_dir):
    """
    تحويل MOT17 GT إلى صيغة YOLO المعيارية:
    output_dir/
      images/train/*.jpg
      labels/train/*.txt   ← class cx cy w h (مُطبَّع 0-1)
      mot17.yaml
    """
    import shutil
    output_dir = Path(output_dir)
    imgs_dir   = output_dir / "images" / "train"
    lbls_dir   = output_dir / "labels" / "train"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    lbls_dir.mkdir(parents=True, exist_ok=True)

    train_path  = Path(mot17_train_path)
    total_imgs  = 0
    total_boxes = 0

    print("\n[FINETUNE] Preparing MOT17 YOLO dataset...")
    print(f"  Source : {train_path}")
    print(f"  Output : {output_dir}")

    sequences = sorted(train_path.iterdir())
    for seq in sequences:
        if not seq.is_dir(): continue
        gt_file = seq / "gt" / "gt.txt"
        img_dir = seq / "img1"
        if not gt_file.exists() or not img_dir.exists(): continue

        # تخطي تسلسلات DPM و FRCNN لتجنب التكرار (SDP فقط)
        if "DPM" in seq.name or "FRCNN" in seq.name:
            continue

        print(f"  Processing {seq.name}...")
        gt = np.loadtxt(str(gt_file), delimiter=",")

        for img_path in sorted(img_dir.glob("*.jpg")):
            fid = int(img_path.stem)

            # فلترة GT:
            # col6 = 1 → not ignored | col7 = 1 → person class
            mask = (gt[:, 0] == fid)
            if gt.shape[1] > 6:
                mask &= (gt[:, 6] == 1)   # not ignored
            if gt.shape[1] > 7:
                mask &= (gt[:, 7] == 1)   # person class
            frame_gt = gt[mask]

            if len(frame_gt) == 0:
                continue

            # قراءة حجم الصورة
            img = cv2.imread(str(img_path))
            if img is None: continue
            H, W = img.shape[:2]

            # فلترة الصناديق الصغيرة جداً
            valid_rows = []
            for row in frame_gt:
                x, y, bw, bh = row[2], row[3], row[4], row[5]
                if bw < 10 or bh < 20: continue   # تجاهل الأشخاص الصغار جداً
                if x < 0 or y < 0: continue
                valid_rows.append(row)

            if not valid_rows:
                continue

            # نسخ الصورة
            dst_name = f"{seq.name}_{img_path.name}"
            dst_img  = imgs_dir / dst_name
            shutil.copy(str(img_path), str(dst_img))

            # كتابة label
            dst_lbl = lbls_dir / f"{seq.name}_{img_path.stem}.txt"
            with open(dst_lbl, "w") as f:
                for row in valid_rows:
                    x, y, bw, bh = row[2], row[3], row[4], row[5]
                    # تطبيع إلى 0-1
                    cx = (x + bw / 2) / W
                    cy = (y + bh / 2) / H
                    nw = bw / W
                    nh = bh / H
                    # تأكد من الحدود
                    cx = float(np.clip(cx, 0, 1))
                    cy = float(np.clip(cy, 0, 1))
                    nw = float(np.clip(nw, 0, 1))
                    nh = float(np.clip(nh, 0, 1))
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    total_boxes += 1

            total_imgs += 1

    # كتابة YAML
    yaml_path = output_dir / "mot17.yaml"
    yaml_content = f"""# MOT17 Fine-tune Dataset
path: {output_dir}
train: images/train
val:   images/train

nc: 1
names:
  0: person
"""
    yaml_path.write_text(yaml_content)

    print(f"  ✅ Dataset ready:")
    print(f"     Images : {total_imgs:,}")
    print(f"     Boxes  : {total_boxes:,}")
    print(f"     YAML   : {yaml_path}")
    return str(yaml_path)


def run_yolo_finetune(yaml_path):
    """
    Fine-tune YOLOv8x على MOT17 train
    يُنتج best.pt في FINETUNE_OUTPUT/mot17/weights/
    """
    from ultralytics import YOLO as UltralyticsYOLO

    print("\n" + "="*70)
    print("🎓  YOLO FINE-TUNE ON MOT17")
    print("="*70)
    print(f"  Base model : {config.FINETUNE_BASE_MODEL}")
    print(f"  Epochs     : {config.FINETUNE_EPOCHS}")
    print(f"  Image size : {config.FINETUNE_IMGSZ}")
    print(f"  Batch      : {config.FINETUNE_BATCH}")
    print(f"  LR         : {config.FINETUNE_LR}")
    print(f"  Output     : {config.FINETUNE_OUTPUT}")
    print(f"  Est. time  : ~{config.FINETUNE_EPOCHS * 2} minutes on GPU")
    print("="*70)

    model = UltralyticsYOLO(config.FINETUNE_BASE_MODEL)

    results = model.train(
        data        = yaml_path,
        epochs      = config.FINETUNE_EPOCHS,
        imgsz       = config.FINETUNE_IMGSZ,
        batch       = config.FINETUNE_BATCH,
        lr0         = config.FINETUNE_LR,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs    = 3,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,
        box         = 7.5,
        cls         = 0.5,
        dfl         = 1.5,
        patience    = config.FINETUNE_PATIENCE,
        device      = 0 if torch.cuda.is_available() else "cpu",
        workers     = 2,
        project     = str(config.FINETUNE_OUTPUT),
        name        = "mot17",
        exist_ok    = True,
        save        = True,
        save_period = 10,
        val         = True,
        plots       = True,
        verbose     = True,
        # augmentation مناسب لـ MOT17
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 0.0,      # لا تدوير (كاميرات ثابتة)
        translate   = 0.1,
        scale       = 0.5,
        shear       = 0.0,
        perspective = 0.0,
        flipud      = 0.0,      # لا قلب عمودي
        fliplr      = 0.5,
        mosaic      = 0.5,      # تقليل mosaic لبيانات MOT
        mixup       = 0.0,
        copy_paste  = 0.1,
    )

    best_weights = config.FINETUNE_OUTPUT / "mot17" / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n  ✅ Fine-tune complete!")
        print(f"  📦 Best weights: {best_weights}")
        return str(best_weights)
    else:
        print(f"\n  ⚠️  best.pt not found, using last.pt")
        last_weights = config.FINETUNE_OUTPUT / "mot17" / "weights" / "last.pt"
        return str(last_weights) if last_weights.exists() else config.FINETUNE_BASE_MODEL


def finetune_pipeline():
    """
    خط أنابيب Fine-tune كامل:
    1. تحضير Dataset
    2. Fine-tune YOLOv8
    3. تحديث config.YOLO_WEIGHTS
    يعيد True إن نجح، False إن فشل أو تخطّى
    """
    if not config.FINETUNE_ENABLED:
        print("[FINETUNE] Disabled — using pretrained weights")
        return False

    # تخطي إن كان النموذج موجوداً
    if config.FINETUNE_SKIP_IF_EXISTS and config.FINETUNE_WEIGHTS.exists():
        print(f"[FINETUNE] Found existing weights: {config.FINETUNE_WEIGHTS}")
        print("[FINETUNE] Skipping training — delete file to retrain")
        config.YOLO_WEIGHTS = str(config.FINETUNE_WEIGHTS)
        return True

    try:
        # الخطوة 1: تحضير Dataset
        mot17_train = config.KAGGLE_MOT17 / "train"
        if not mot17_train.exists():
            print(f"[FINETUNE] Train path not found: {mot17_train}")
            return False

        yaml_path = prepare_mot17_yolo_dataset(
            str(mot17_train),
            str(config.FINETUNE_DATASET)
        )

        # الخطوة 2: Fine-tune
        best_weights = run_yolo_finetune(yaml_path)

        # الخطوة 3: تحديث YOLO_WEIGHTS
        config.YOLO_WEIGHTS = best_weights
        print(f"\n[FINETUNE] ✅ Using fine-tuned weights: {best_weights}")
        return True

    except Exception as e:
        print(f"\n[FINETUNE] ❌ Failed: {e}")
        print("[FINETUNE] Falling back to pretrained weights")
        import traceback
        traceback.print_exc()
        return False


def print_finetune_results():
    """عرض نتائج التدريب من ملفات Ultralytics"""
    results_csv = config.FINETUNE_OUTPUT / "mot17" / "results.csv"
    if not results_csv.exists():
        return

    try:
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        if not rows: return

        last = rows[-1]
        best_epoch = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0))

        print("\n" + "─"*60)
        print("📈  FINE-TUNE TRAINING RESULTS")
        print("─"*60)
        print(f"  Total epochs    : {len(rows)}")
        print(f"  Best epoch      : {best_epoch.get('epoch','?')}")
        mAP50  = float(best_epoch.get("metrics/mAP50(B)",  0) or 0) * 100
        mAP50_95 = float(best_epoch.get("metrics/mAP50-95(B)", 0) or 0) * 100
        box_loss = float(last.get("train/box_loss", 0) or 0)
        print(f"  Best mAP@0.5    : {mAP50:.1f}%")
        print(f"  Best mAP@0.5:95 : {mAP50_95:.1f}%")
        print(f"  Final box loss  : {box_loss:.4f}")
        if mAP50 >= 85:
            print("  ✅ Excellent fine-tune quality!")
        elif mAP50 >= 75:
            print("  🟡 Good — consider more epochs for better results")
        else:
            print("  🔴 Low mAP — check dataset or increase epochs")
        print("─"*60)
    except Exception as e:
        pass  # نتجاهل أخطاء القراءة



def load_yolo():
    weights = str(config.YOLO_WEIGHTS)
    source  = "fine-tuned" if "finetune" in weights or "best.pt" in weights else "pretrained"
    print(f"[INFO] Loading YOLO ({source}): {weights}")
    model = YOLO(weights)
    model.to(config.TORCH_DEVICE)
    return model

def load_frcnn():
    if config.SKIP_FRCNN:
        print("[INFO] Faster R-CNN skipped (FAST_MODE)")
        return None
    print("[INFO] Loading Faster R-CNN (ResNet-50 FPN) on GPU...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval().to(config.TORCH_DEVICE)
    return model

def run_yolo(yolo_model, img_folder, total_frames, yolo_conf):
    images = sorted(img_folder.glob('*.jpg'))
    if config.MAX_FRAMES:
        images = images[:config.MAX_FRAMES]
    yolo_dets = []
    img_size  = 640 if config.FAST_MODE else 1280
    batch_size = config.FAST_BATCH if config.FAST_MODE else (12 if torch.cuda.is_available() else 4)
    for i in tqdm(range(0, len(images), batch_size), desc="  YOLO", ncols=70):
        batch = images[i:i+batch_size]
        results = yolo_model([str(p) for p in batch], conf=yolo_conf,
                             iou=config.YOLO_IOU, classes=[0], verbose=False,
                             device=config.DEVICE, imgsz=img_size,
                             half=config.FP16 and torch.cuda.is_available())
        for r in results:
            if len(r.boxes)>0:
                boxes=r.boxes.xyxy.cpu().numpy(); scores=r.boxes.conf.cpu().numpy()
                yolo_dets.append(np.column_stack([boxes,scores,np.zeros(len(boxes))]))
            else:
                yolo_dets.append(np.array([]))
    while len(yolo_dets)<total_frames: yolo_dets.append(np.array([]))
    return yolo_dets[:total_frames]

@torch.no_grad()
def run_frcnn_gpu(frcnn_model, img_folder, total_frames, frcnn_conf=None):
    # ── FAST MODE: تخطي Faster R-CNN لتوفير 60% من وقت التشغيل ─────────────
    if config.SKIP_FRCNN:
        print("  [FAST] Faster R-CNN skipped — YOLO-only mode")
        return [np.zeros((0, 6), dtype=np.float32)] * total_frames
    # ────────────────────────────────────────────────────────────────────────
    images = sorted(img_folder.glob('*.jpg'))
    if config.MAX_FRAMES:
        images = images[:config.MAX_FRAMES]
    frcnn_dets = []
    det_thr = config.FRCNN_CONF if frcnn_conf is None else float(frcnn_conf)
    for img_path in tqdm(images, desc="  FRCNN", ncols=70):
        img=cv2.imread(str(img_path))
        if img is None:
            frcnn_dets.append(np.array([])); continue
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        tensor=torchvision.transforms.functional.to_tensor(img_rgb).to(config.TORCH_DEVICE)
        out=frcnn_model([tensor])[0]
        dets=[]
        for box,score,label in zip(out["boxes"].cpu().numpy(),out["scores"].cpu().numpy(),out["labels"].cpu().numpy()):
            if label!=1 or score<det_thr: continue
            x1,y1,x2,y2=float(box[0]),float(box[1]),float(box[2]),float(box[3])
            x1,y1=max(1.0,x1),max(1.0,y1); x2,y2=max(x1+1,x2),max(y1+1,y2)
            dets.append([x1,y1,x2,y2,float(score),0.0])
        frcnn_dets.append(np.array(dets) if dets else np.array([]))
    while len(frcnn_dets)<total_frames: frcnn_dets.append(np.array([]))
    return frcnn_dets[:total_frames]


# ==========================================================================================
# DETECTOR FUSION — FIS حقيقي لدمج YOLO + FRCNN
# ==========================================================================================

def fis_detector_fusion(
    yolo_dets, frcnn_dets, single_detector_penalty=0.72, block_frcnn_only=False, yolo_only_min_conf=0.72
):
    """
    دمج ضبابي حقيقي باستخدام DetectorFusionFIS
    
    لكل كشف في YOLO، نبحث عن أقرب كشف في FRCNN (IoU > 0.3)
    ثم نستخدم FIS لحساب درجة الثقة المدموجة
    """
    all_dets = []
    for det in (yolo_dets if len(yolo_dets)>0 else []):
        all_dets.append({'box':np.array(det[:4]),'conf':float(det[4]),'src':'yolo'})
    for det in (frcnn_dets if len(frcnn_dets)>0 else []):
        all_dets.append({'box':np.array(det[:4]),'conf':float(det[4]),'src':'frcnn'})

    if not all_dets: return np.array([])

    fused=[]; used=set()

    for i,d1 in enumerate(all_dets):
        if i in used: continue
        group=[d1]; used.add(i)

        for j,d2 in enumerate(all_dets):
            if j in used: continue
            if compute_iou(d1['box'],d2['box'])>0.25:
                group.append(d2); used.add(j)

        sources=set(d['src'] for d in group)
        boxes=np.array([d['box'] for d in group])
        confs=np.array([d['conf'] for d in group])

        if len(sources)==2:
            # كاشفان متفقان → استخدم FIS
            best_iou = -1.0
            best_pair = None
            for gi in range(len(group)):
                for gj in range(gi+1, len(group)):
                    if group[gi]['src'] == group[gj]['src']:
                        continue
                    pair_iou = compute_iou(group[gi]['box'], group[gj]['box'])
                    if pair_iou > best_iou:
                        best_iou = pair_iou
                        best_pair = (group[gi], group[gj])
            if best_pair is None:
                continue
            iou_score = best_iou
            conf_avg  = 0.5 * (float(best_pair[0]['conf']) + float(best_pair[1]['conf']))
            fusion_conf = TRACKER_FIS_ENGINE.detector_fis.compute(iou_score, conf_avg)
        else:
            max_conf = float(np.max(confs))
            source = next(iter(sources))
            if source == "frcnn" and block_frcnn_only:
                continue
            if source == "yolo":
                if max_conf < float(yolo_only_min_conf):
                    continue
                fusion_conf = max_conf * 0.92
            else:
                fusion_conf = max_conf * float(single_detector_penalty)

        weights=confs/(confs.sum()+1e-9)
        fused_box=np.average(boxes,axis=0,weights=weights)
        fused.append([*fused_box, float(np.clip(fusion_conf,0,1)), 0.0])

    return np.array(fused) if fused else np.array([])


# ==========================================================================================
# TRACKER FUSION — FIS حقيقي لدمج ByteTrack + BoT-SORT
# ==========================================================================================

def fis_tracker_fusion(byte_tracks, bot_tracks, memory_stabilizer=None, warp_matrix=None, fusion_cfg=None):
    """دمج ضبابي حقيقي للمتتبعَين إطاراً بإطار"""
    merged=[]; used_bot=set()
    byte_tracks=list(byte_tracks) if byte_tracks is not None else []
    bot_tracks =list(bot_tracks)  if bot_tracks  is not None else []
    fusion_cfg=fusion_cfg or {}
    density=max(len(byte_tracks),len(bot_tracks))
    keep_iou=float(fusion_cfg.get("KEEP_IOU",0.58))
    recovery_iou=float(fusion_cfg.get("RECOVERY_IOU",0.60))
    switch_penalty=float(fusion_cfg.get("SWITCH_PENALTY",0.88))
    stable_merge_margin=float(fusion_cfg.get("STABLE_MERGE_MARGIN",0.02))
    unmatched_bot_conf_min=float(fusion_cfg.get("UNMATCHED_BOT_CONF_MIN",0.18))
    bot_overlap_gate=float(fusion_cfg.get("BOT_OVERLAP_GATE",0.30))
    stable_weight_boost=float(fusion_cfg.get("STABLE_WEIGHT_BOOST",1.15))
    adaptive_match_min_thr=float(fusion_cfg.get("ADAPTIVE_MATCH_MIN_THR",0.55))
    adaptive_match_max_thr=float(fusion_cfg.get("ADAPTIVE_MATCH_MAX_THR",0.75))
    merge_thr=compute_adaptive_merge_threshold(density,base_thr=float(fusion_cfg.get("MERGE_BASE_THR",0.57)))

    for bt in byte_tracks:
        bt=np.array(bt,dtype=np.float32).flatten()
        if len(bt)<5: continue
        bt_box=bt[:4].copy(); bt_cmp=bt_box.copy()
        if warp_matrix is not None:
            bt_cmp=CameraMotionCompensator.compensate_box(bt_box,warp_matrix)

        bt_tid=int(bt[4]); conf_bt=float(bt[5]) if len(bt)>5 else 0.5
        bt_prev=memory_stabilizer.get_previous_box(bt_tid) if memory_stabilizer else None
        bt_continuity=compute_iou(bt_prev,bt_cmp) if bt_prev is not None else 0.0
        bt_is_stable=bool(memory_stabilizer and memory_stabilizer.was_matched_last_frame(bt_tid) and bt_continuity>=keep_iou)

        best_match=None; best_score=0.0; best_result=None

        for i,bot in enumerate(bot_tracks):
            if i in used_bot: continue
            bot=np.array(bot,dtype=np.float32).flatten()
            if len(bot)<5: continue
            bot_box=bot[:4].copy(); bot_cmp=bot_box.copy()
            if warp_matrix is not None:
                bot_cmp=CameraMotionCompensator.compensate_box(bot_box,warp_matrix)

            conf_bot=float(bot[5]) if len(bot)>5 else 0.5
            bot_prev=memory_stabilizer.get_previous_box(int(bot[4])) if memory_stabilizer else None
            if bot_prev is None: bot_prev=bot_cmp

            iou       =compute_iou(bt_cmp,bot_cmp)
            motion_sim=compute_motion_similarity(bt_cmp,bot_cmp)
            if iou<0.12 and motion_sim<0.15: continue

            areaA=max(1.0,(bt_cmp[2]-bt_cmp[0])*(bt_cmp[3]-bt_cmp[1]))
            areaB=max(1.0,(bot_cmp[2]-bot_cmp[0])*(bot_cmp[3]-bot_cmp[1]))
            size_ratio=min(areaA,areaB)/max(areaA,areaB)

            diagA=math.sqrt((bt_cmp[2]-bt_cmp[0])**2+(bt_cmp[3]-bt_cmp[1])**2)
            diagB=math.sqrt((bot_cmp[2]-bot_cmp[0])**2+(bot_cmp[3]-bot_cmp[1])**2)
            c1x=(bt_cmp[0]+bt_cmp[2])/2; c1y=(bt_cmp[1]+bt_cmp[3])/2
            c2x=(bot_cmp[0]+bot_cmp[2])/2; c2y=(bot_cmp[1]+bot_cmp[3])/2
            center_dist=math.sqrt((c1x-c2x)**2+(c1y-c2y)**2)
            center_sim=1.0/(1.0+center_dist/(max(diagA,diagB)+1e-9))
            vel_dir=compute_velocity_direction_similarity(bt_prev,bt_cmp,bot_prev,bot_cmp)
            conf_consist=float(np.clip(1.0-abs(conf_bt-conf_bot),0.0,1.0))
            history_bt =compute_history_consistency(bt_cmp,bt_prev)
            history_bot=compute_history_consistency(bot_cmp,bot_prev)
            history_consist=float(np.clip(1.0-abs(history_bt-history_bot),0.0,1.0))

            fis_result=TRACKER_FIS_ENGINE.compute_tracker(
                iou,center_sim,size_ratio,motion_sim,vel_dir,conf_consist,history_consist)
            history_mean=0.5*(history_bt+history_bot)
            fuzzy_score=0.80*fis_result["final_score"]+0.20*history_mean

            if bot[4]!=bt_tid and bt_continuity>=max(keep_iou,0.58):
                fuzzy_score*=switch_penalty
            if memory_stabilizer and memory_stabilizer.was_matched_last_frame(bt_tid):
                fuzzy_score=min(1.0,fuzzy_score*1.08)  # stronger boost for known tracks
            if bt_prev is not None:
                prev_cx=(bt_prev[0]+bt_prev[2])/2; prev_cy=(bt_prev[1]+bt_prev[3])/2
                jump=math.sqrt((c1x-prev_cx)**2+(c1y-prev_cy)**2)
                if jump>3.0*diagA: fuzzy_score*=0.78  # less aggressive penalty

            if best_result is None or fuzzy_score>best_score:
                best_score=float(np.clip(fuzzy_score,0,1)); best_match=i; best_result=fis_result

        accept_thr=merge_thr+(stable_merge_margin if bt_is_stable else 0.0)
        if best_match is not None:
            prox = best_result["proximity_score"] if best_result else 0.5
            accept_thr=max(
                accept_thr,
                compute_adaptive_match_threshold(
                    prox, min_thr=adaptive_match_min_thr, max_thr=adaptive_match_max_thr
                ),
            )
        if best_match is not None and best_score>=accept_thr:
            bot=np.array(bot_tracks[best_match],dtype=np.float32).flatten()
            conf_bot=float(bot[5]) if len(bot)>5 else 0.5
            w1=conf_bt*(stable_weight_boost if bt_is_stable else 1.0); w2=conf_bot; s=w1+w2+1e-9
            merged_box=bt[:4]*(w1/s)+bot[:4]*(w2/s)
            merged_conf=float(np.clip(max(conf_bt,conf_bot,best_score)*1.02,0,1))  # slight boost for matched pairs
            merged_tid=int(bt[4])
            if memory_stabilizer:
                keep=memory_stabilizer.should_keep_current_id(merged_tid,merged_box,continuity_iou=keep_iou)
                if not keep:
                    prev_tid,prev_iou=memory_stabilizer.get_best_previous_match(
                        merged_box,min_iou=recovery_iou,exclude_tid=merged_tid,only_lost=True)
                    if prev_tid is not None and prev_iou>=max(recovery_iou+0.02,0.62) and memory_stabilizer.can_reassign(merged_tid):
                        memory_stabilizer.note_reassignment(merged_tid,prev_tid)
                        merged_tid=int(prev_tid)
            merged.append(np.array([*merged_box,merged_tid,merged_conf],dtype=np.float32))
            used_bot.add(best_match)
            continue

        fallback=bt[:6] if len(bt)>=6 else np.array([*bt[:5],0.5],dtype=np.float32)
        if memory_stabilizer:
            keep=memory_stabilizer.should_keep_current_id(int(fallback[4]),fallback[:4],continuity_iou=keep_iou)
            if not keep:
                prev_tid,prev_iou=memory_stabilizer.get_best_previous_match(
                    fallback[:4],min_iou=max(recovery_iou,0.60),exclude_tid=int(fallback[4]),only_lost=True)
                if prev_tid is not None and prev_iou>=max(recovery_iou+0.01,0.62) and memory_stabilizer.can_reassign(int(fallback[4])):
                    memory_stabilizer.note_reassignment(int(fallback[4]),prev_tid)
                    fallback[4]=int(prev_tid)
        merged.append(fallback)

    for i,bot in enumerate(bot_tracks):
        if i in used_bot: continue
        bot=np.array(bot,dtype=np.float32).flatten()
        if len(bot)>=5:
            conf_bot=float(bot[5]) if len(bot)>5 else 0.5
            if conf_bot<unmatched_bot_conf_min: continue
            if any(compute_iou(bot[:4],m[:4])>=bot_overlap_gate for m in merged): continue
            fallback=bot[:6] if len(bot)>=6 else np.array([*bot[:5],0.5],dtype=np.float32)
            if memory_stabilizer:
                prev_tid,prev_iou=memory_stabilizer.get_best_previous_match(
                    fallback[:4],min_iou=max(recovery_iou+0.02,0.62),exclude_tid=int(fallback[4]),only_lost=True)
                if prev_tid is not None and prev_iou>=max(recovery_iou+0.03,0.65):
                    fallback[4]=int(prev_tid)
            merged.append(fallback)

    if memory_stabilizer:
        for t in merged:
            if len(t)>=6: memory_stabilizer.update_track(int(t[4]),t[:4],float(t[5]))

    return merged


# ==========================================================================================
# POST-PROCESSING
# ==========================================================================================

def interpolate_tracks(tracks, max_gap=60):
    by_id=defaultdict(dict)
    for fi,ft in enumerate(tracks):
        for t in ft:
            if len(t)>=5: by_id[int(t[4])][fi]=t.copy()
    total=len(tracks); result=[[] for _ in range(total)]
    for tid,fd in by_id.items():
        frames=sorted(fd.keys())
        for k in range(len(frames)-1):
            fa,fb=frames[k],frames[k+1]; gap=fb-fa-1
            result[fa].append(fd[fa])
            ta,tb=fd[fa],fd[fb]
            continuity=compute_history_consistency(tb[:4],ta[:4])
            effective_gap=max_gap+(15 if continuity>=0.55 else (5 if continuity>=0.40 else 0))
            if 0<gap<=effective_gap:
                for fi in range(fa+1,fb):
                    a=(fi-fa)/(fb-fa)
                    result[fi].append(np.array([*(1-a)*ta[:4]+a*tb[:4],tid,(1-a)*float(ta[5])+a*float(tb[5])],dtype=np.float32))
        if frames: result[frames[-1]].append(fd[frames[-1]])
    return result

def remove_short_tracks(tracks, min_length=2):
    counts=defaultdict(int)
    for ft in tracks:
        for t in ft: counts[int(t[4])]+=1
    return [[t for t in ft if counts[int(t[4])]>=min_length] for ft in tracks]

def smooth_trajectories(tracks):
    smoother=TrajectorySmoother(max_hist=7); prev_centers={}; smoothed=[]
    for ft in tracks:
        nf=[]
        for t in ft:
            if len(t)<6: continue
            tid=int(t[4]); conf=float(t[5])
            cx,cy=(t[0]+t[2])/2,(t[1]+t[3])/2
            speed=None
            if tid in prev_centers:
                pcx,pcy=prev_centers[tid]; speed=math.sqrt((cx-pcx)**2+(cy-pcy)**2)
            prev_centers[tid]=(cx,cy)
            sb=smoother.smooth(tid,t[:4],speed=speed)
            nf.append(np.array([*sb,tid,conf],dtype=np.float32))
        smoothed.append(nf)
    return smoothed

def final_clean(tracks, img_w=1920, img_h=1080, min_track_conf=None):
    min_track_conf=config.MIN_TRACK_CONF if min_track_conf is None else float(min_track_conf)
    cleaned=[]
    for ft in tracks:
        fc=[]
        for t in ft:
            if len(t)<6: continue
            x1,y1,x2,y2,tid,conf=t[:6]
            x1=max(1.0,min(float(x1),img_w-2)); y1=max(1.0,min(float(y1),img_h-2))
            x2=max(x1+1,min(float(x2),img_w-1)); y2=max(y1+1,min(float(y2),img_h-1))
            if (x2-x1)<16 or (y2-y1)<32: continue
            conf=float(np.clip(conf,0,1))
            if conf<min_track_conf: continue
            if not all(is_valid_number(v) for v in [x1,y1,x2,y2,tid,conf]): continue
            fc.append(np.array([x1,y1,x2,y2,int(tid),conf],dtype=np.float32))
        cleaned.append(fc)
    return cleaned


# ==========================================================================================
# OUTPUT — ZIP STRUCTURE (من هيكل V11 الناجح)
# ==========================================================================================

def remove_duplicate_tracks(tracks, iou_thr=0.85):
    """
    إزالة المسارات المكررة (Fragmentation) بناءً على التداخل العالي
    مسارات مختلفة IDs تتداخل بشكل كبير → دمج الأقصر في الأطول
    """
    from collections import defaultdict
    # بناء خريطة: frame_idx → قائمة tracks
    by_id = defaultdict(dict)
    for fi, ft in enumerate(tracks):
        for t in ft:
            if len(t) >= 5:
                by_id[int(t[4])][fi] = t

    id_lengths = {tid: len(frames) for tid, frames in by_id.items()}
    merge_map = {}  # from_tid → to_tid

    ids = list(by_id.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            tidA, tidB = ids[i], ids[j]
            if tidA in merge_map or tidB in merge_map:
                continue
            framesA = set(by_id[tidA].keys())
            framesB = set(by_id[tidB].keys())
            common = framesA & framesB
            if len(common) < 3:
                continue
            overlap_ious = []
            for f in list(common)[:20]:
                iou = compute_iou(by_id[tidA][f][:4], by_id[tidB][f][:4])
                overlap_ious.append(iou)
            if np.mean(overlap_ious) >= iou_thr:
                # دمج المسار الأقصر في الأطول
                if id_lengths[tidA] >= id_lengths[tidB]:
                    merge_map[tidB] = tidA
                else:
                    merge_map[tidA] = tidB

    if not merge_map:
        return tracks

    # تطبيق الدمج
    result = []
    for fi, ft in enumerate(tracks):
        new_ft = []
        for t in ft:
            if len(t) >= 5:
                tid = int(t[4])
                resolved = tid
                while resolved in merge_map:
                    resolved = merge_map[resolved]
                t = t.copy(); t[4] = resolved
            new_ft.append(t)
        result.append(new_ft)
    return result



def save_mot_txt(tracks, out_path):
    """حفظ نتائج التتبع بصيغة MOTChallenge القياسية"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for fi, ft in enumerate(tracks):
        fn = fi + 1
        for t in ft:
            if len(t) < 6: continue
            x1, y1, x2, y2, tid, conf = t[:6]
            if not all(is_valid_number(v) for v in [x1, y1, x2, y2, tid, conf]): continue
            w = x2 - x1; h = y2 - y1
            if w < 1 or h < 1: continue
            conf = float(np.clip(conf, 0, 1))
            lines.append(f"{int(fn)},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")
    with open(out_path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Saved {out_path.name}  ({len(lines)} rows)")


def save_motchallenge_structure(base_seq_id, tracks, output_dir):
    """
    حفظ النتائج بالهيكل الرسمي لـ MOTChallenge:
    output_dir/
      MOT17-{id}-DPM/
        MOT17-{id}-DPM.txt    (داخل مجلد الاسم)
      MOT17-{id}-FRCNN/
        MOT17-{id}-FRCNN.txt
      MOT17-{id}-SDP/
        MOT17-{id}-SDP.txt
    وأيضاً نسخة مسطحة للـ motmetrics:
      output_dir/MOT17-{id}-SDP.txt
    """
    produced = []
    for det in MOT17_DETECTORS:
        name     = f"MOT17-{base_seq_id}-{det}"
        # ── 1. ملف مسطح (للتقييم المحلي بـ motmetrics) ──────────────────────
        flat_path = Path(output_dir) / f"{name}.txt"
        save_mot_txt(tracks, flat_path)
        # ── 2. هيكل MOTChallenge الرسمي ──────────────────────────────────────
        seq_folder = Path(output_dir) / "data" / name
        seq_folder.mkdir(parents=True, exist_ok=True)
        nested_path = seq_folder / f"{name}.txt"
        save_mot_txt(tracks, nested_path)
        produced.append(name)
    return produced



def save_empty_mot_txt(out_path):
    with open(out_path,"w",newline="\n") as f: f.write("")

def generate_all_42_files(base_seq_id, tracks, output_dir):
    return save_motchallenge_structure(base_seq_id, tracks, output_dir)

def ensure_all_42_placeholders(output_dir):
    for name in MOT17_ALL_42:
        # مسطح
        p = Path(output_dir) / f"{name}.txt"
        if not p.exists(): save_empty_mot_txt(p)
        # متداخل (MOTChallenge رسمي)
        nested = Path(output_dir) / "data" / name / f"{name}.txt"
        if not nested.exists():
            nested.parent.mkdir(parents=True, exist_ok=True)
            save_empty_mot_txt(nested)

def build_submission_zip(output_dir, zip_path):
    """
    بناء ZIP بالهيكلَين:
      data/MOT17-xx-DET/MOT17-xx-DET.txt  ← الصيغة الرسمية لـ MOTChallenge
      MOT17-xx-DET.txt                     ← الصيغة المسطحة احتياطياً
    """
    output_dir = Path(output_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        count = 0
        data_dir = output_dir / "data"
        if data_dir.exists():
            for txt in sorted(data_dir.rglob("*.txt")):
                arcname = str(txt.relative_to(output_dir))
                zf.write(txt, arcname=arcname)
                count += 1
        for txt in sorted(output_dir.glob("*.txt")):
            zf.write(txt, arcname=txt.name)
            count += 1
    print(f"[INFO] ZIP: {zip_path}  ({count} files)")

def find_sequence_folder(root_dir, base_seq_id):
    for suffix in ["", "-DPM", "-FRCNN", "-SDP"]:
        p = Path(root_dir) / f"MOT17-{base_seq_id}{suffix}"
        if p.is_dir() and (p / "img1").exists():
            return p
    return None

def verify_zip(zip_path):
    try:
        with zipfile.ZipFile(zip_path,"r") as zf: return zf.testzip() is None
    except: return False


# ==========================================================================================
# SEQUENCE PROCESSING
# ==========================================================================================

def process_sequence(seq_dir, yolo_model, frcnn_model):
    seq_name=seq_dir.name; base_seq_id=seq_name.split('-')[1]
    key=f"MOT17-{base_seq_id}"; spec=Config.get_seq_config(seq_name)

    img_dir=seq_dir/"img1"
    if not img_dir.exists(): return None
    img_files=sorted(img_dir.glob("*.jpg"))
    if not img_files: return None
    first=cv2.imread(str(img_files[0]))
    if first is None: return None
    img_h,img_w=first.shape[:2]; total_frames=len(img_files)

    yolo_conf  =spec.get("YOLO_CONF",  config.YOLO_CONF)
    strict_mota_mode=bool(spec.get("STRICT_MOTA_MODE", False))
    use_temporal_birth_control = key in {"MOT17-08", "MOT17-12"}
    if strict_mota_mode:
        track_buffer=spec.get("TRACK_BUFFER", config.BYTETRACK_TRACK_BUFFER)
    else:
        track_buffer=max(spec.get("TRACK_BUFFER",config.BYTETRACK_TRACK_BUFFER),
                         240 if total_frames<750 else config.BYTETRACK_TRACK_BUFFER)
    match_thresh=spec.get("MATCH_THRESH",config.BYTETRACK_MATCH_THRESH)
    if strict_mota_mode:
        interp_gap=spec.get("INTERPOLATION_MAX_GAP", config.INTERPOLATION_MAX_GAP)
    else:
        interp_gap =max(spec.get("INTERPOLATION_MAX_GAP",config.INTERPOLATION_MAX_GAP),
                        65 if total_frames<750 else config.INTERPOLATION_MAX_GAP)
    min_len    =spec.get("MIN_TRACK_LENGTH",config.MIN_TRACK_LENGTH)
    min_track_conf=spec.get("MIN_TRACK_CONF",config.MIN_TRACK_CONF)
    frcnn_conf=spec.get("FRCNN_CONF", config.FRCNN_CONF)
    track_thresh=spec.get("TRACK_THRESH", config.BYTETRACK_TRACK_THRESH)
    recovery_iou=spec.get("RECOVERY_IOU",0.60)
    keep_iou=spec.get("KEEP_IOU",0.58)
    switch_cooldown=spec.get("SWITCH_COOLDOWN",12)
    memory_min_hits=spec.get("MEMORY_MIN_HITS",3)
    unmatched_bot_conf_min=spec.get("UNMATCHED_BOT_CONF_MIN",max(min_track_conf+0.08,0.18))
    switch_penalty=spec.get("SWITCH_PENALTY",0.88)
    stable_merge_margin=spec.get("STABLE_MERGE_MARGIN",0.02)
    bot_overlap_gate=spec.get("BOT_OVERLAP_GATE",0.30)
    merge_base_thr=spec.get("MERGE_BASE_THR",0.57)
    topk_dets=spec.get("TOPK_DETS", config.TOPK_DETS)
    pre_filter_min_conf=spec.get("PRE_FILTER_MIN_CONF", None)
    fis_conf_gate=spec.get("FIS_CONF_GATE", None)
    block_frcnn_only=bool(spec.get("BLOCK_FRCNN_ONLY", False))
    yolo_only_min_conf=spec.get("YOLO_ONLY_MIN_CONF", 0.80)
    single_detector_penalty=spec.get("SINGLE_DETECTOR_PENALTY", 0.50 if use_temporal_birth_control else 0.70)
    temporal_confirm_iou=spec.get("TEMPORAL_CONFIRM_IOU", 0.50)
    temporal_birth_min_conf=spec.get("TEMPORAL_BIRTH_MIN_CONF", 0.72)
    adaptive_match_min_thr=spec.get("ADAPTIVE_MATCH_MIN_THR",0.55)
    adaptive_match_max_thr=spec.get("ADAPTIVE_MATCH_MAX_THR",0.75)

    print(f"  yolo_conf={yolo_conf}  buffer={track_buffer}  match={match_thresh}  interp={interp_gap}  min_conf={min_track_conf}")

    # 1. YOLO
    print("[1/5] YOLO..."); yolo_dets=run_yolo(yolo_model,img_dir,total_frames,yolo_conf)
    print(f"  {sum(len(d) for d in yolo_dets):,} كشف")

    # 2. Faster R-CNN على GPU
    print("[2/5] Faster R-CNN (GPU)..."); frcnn_dets=run_frcnn_gpu(frcnn_model,img_dir,total_frames,frcnn_conf=frcnn_conf)
    print(f"  {sum(len(d) for d in frcnn_dets):,} كشف")

    # 3. دمج ضبابي حقيقي للكواشف
    print("[3/5] FIS دمج الكواشف (YOLO + FRCNN)...")
    fused_dets=[
        fis_detector_fusion(
            y, f,
            single_detector_penalty=single_detector_penalty,
            block_frcnn_only=block_frcnn_only,
            yolo_only_min_conf=yolo_only_min_conf,
        )
        for y,f in zip(yolo_dets,frcnn_dets)
    ]
    if fis_conf_gate is not None:
        gated_fused = []
        for dets in fused_dets:
            if len(dets) > 0:
                density = len(dets)
                # تخفيف بوابة الثقة — نحتفظ بالكشوفات المتوسطة لتقليل FN
                extra_gate = 0.03 if density > 50 else (0.01 if density > 30 else 0.0)
                dets = dets[dets[:,4] >= float(fis_conf_gate + extra_gate)]
            gated_fused.append(dets)
        fused_dets = gated_fused
    if strict_mota_mode:
        fused_dets=[
            pre_filter_fused_detections(d, topk_dets=topk_dets, min_conf_override=pre_filter_min_conf)
            for d in fused_dets
        ]
    print(f"  {sum(len(d) for d in fused_dets):,} كشف مدموج")

    # 4. التتبع
    print("[4/5] ByteTrack + BoT-SORT...")
    byte_tracker=ByteTrack(
        min_conf=track_thresh,
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        track_buffer=track_buffer,
        frame_rate=30
    )
    try:
        reid_path = str(config.REID_WEIGHTS) if Path(config.REID_WEIGHTS).exists() else None
        bot_tracker = BoTSORT(
            reid_model=reid_path,
            track_high_thresh=track_thresh,
            track_low_thresh=max(0.05, track_thresh - 0.10),
            new_track_thresh=track_thresh + 0.05,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method='ecc',
            frame_rate=30,
            with_reid=(reid_path is not None),
        )
        use_bot=True
    except Exception as e:
        print(f"  ⚠️ BoT-SORT: {e}"); use_bot=False; bot_tracker=None

    memory_age=spec.get("MEMORY_MAX_AGE", max(100,min(160,track_buffer//2+interp_gap//3)))
    memory_stabilizer=TrackMemoryStabilizer(
        max_age=memory_age,recovery_iou=recovery_iou,min_hits=memory_min_hits,switch_cooldown=switch_cooldown)
    cmc=CameraMotionCompensator()
    final_tracks=[]
    prev_birth_dets=np.zeros((0, 6), dtype=np.float32)
    fusion_cfg={"KEEP_IOU":keep_iou,"RECOVERY_IOU":recovery_iou,
                "UNMATCHED_BOT_CONF_MIN":unmatched_bot_conf_min,"SWITCH_PENALTY":switch_penalty,
                "STABLE_MERGE_MARGIN":stable_merge_margin,"BOT_OVERLAP_GATE":bot_overlap_gate,
                "MERGE_BASE_THR":merge_base_thr,
                "ADAPTIVE_MATCH_MIN_THR":adaptive_match_min_thr,
                "ADAPTIVE_MATCH_MAX_THR":adaptive_match_max_thr,
                "STABLE_WEIGHT_BOOST":1.20}

    for raw_dets,img_path in zip(fused_dets,img_files):
        dets=raw_dets
        if use_temporal_birth_control and len(raw_dets) > 0:
            confirmed = []
            for det in raw_dets:
                if has_recent_track_support(det[:4], memory_stabilizer, existing_iou=0.35):
                    confirmed.append(det)
                    continue
                temp = temporally_confirm_detections(
                    np.array([det], dtype=np.float32),
                    prev_birth_dets,
                    memory_stabilizer=memory_stabilizer,
                    iou_thr=temporal_confirm_iou,
                    high_conf_pass=temporal_birth_min_conf,
                )
                if len(temp) > 0:
                    confirmed.append(det)
            dets=np.array(confirmed, dtype=np.float32) if confirmed else np.zeros((0, 6), dtype=np.float32)
            prev_birth_dets=dets.copy() if len(dets) > 0 else np.zeros((0, 6), dtype=np.float32)
        elif use_temporal_birth_control:
            prev_birth_dets=np.zeros((0, 6), dtype=np.float32)
        img=cv2.imread(str(img_path))
        warp=cmc.estimate(img) if img is not None else None

        try:
            bt=byte_tracker.update(dets) if len(dets)>0 else []
        except Exception as e:
            try:
                bt=byte_tracker.update(dets, None) if len(dets)>0 else []
            except:
                bt=[]

        if use_bot and bot_tracker and img is not None and len(dets)>0:
            try: bo=bot_tracker.update(dets,img)
            except: bo=[]
        else: bo=[]

        memory_stabilizer.step()

        # 5. دمج ضبابي حقيقي للمتتبعات
        # تحويل مخرجات boxmot الجديدة → [x1,y1,x2,y2,tid,conf]
        def parse_tracks(raw):
            result = []
            if raw is None: return result
            for t in raw:
                t = np.array(t, dtype=np.float32).flatten()
                if len(t) >= 6:
                    result.append(t[:6])   # x1,y1,x2,y2,tid,conf
                elif len(t) == 5:
                    result.append(np.array([*t[:5], 0.5], dtype=np.float32))
            return result

        bt_parsed = parse_tracks(bt)
        bo_parsed = parse_tracks(bo)

        merged=fis_tracker_fusion(bt_parsed,bo_parsed,memory_stabilizer,warp,fusion_cfg=fusion_cfg)
        fr=[np.array([x1,y1,x2,y2,int(tid),float(conf)],dtype=np.float32)
            for t in merged
            for x1,y1,x2,y2,tid,conf in [np.array(t,dtype=np.float32).flatten()[:6]]
            if len(t)>=6 and float(conf)>=min_track_conf]
        final_tracks.append(fr)

    print("[5/5] Post-processing...")
    final_tracks=final_clean(final_tracks,img_w=img_w,img_h=img_h,min_track_conf=min_track_conf)
    if config.USE_INTERPOLATION:
        final_tracks=interpolate_tracks(final_tracks,max_gap=interp_gap)
    final_tracks=smooth_trajectories(final_tracks)
    final_tracks=final_clean(final_tracks,img_w=img_w,img_h=img_h,min_track_conf=min_track_conf)
    final_tracks=remove_short_tracks(final_tracks,min_length=min_len)
    final_tracks=remove_duplicate_tracks(final_tracks,iou_thr=0.85)  # إزالة المسارات المكررة
    final_tracks=final_clean(final_tracks,img_w=img_w,img_h=img_h,min_track_conf=min_track_conf)

    return base_seq_id, final_tracks


# ==========================================================================================
# EVALUATION — motmetrics
# ==========================================================================================

def evaluate_results(gt_root, pred_dir):
    """تقييم محلي كامل مع إصلاح توافق NumPy 2.0"""
    import subprocess, sys

    # ── إصلاح مشكلة np.asfarray في NumPy 2.0 ────────────────────────────────
    import numpy as np
    if not hasattr(np, 'asfarray'):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

    try:
        import motmetrics as mm
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "motmetrics", "-q"])
        import motmetrics as mm

    # تطبيق الإصلاح داخل motmetrics أيضاً
    try:
        import motmetrics.math_util as mu
        if not hasattr(np, 'asfarray'):
            np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
    except: pass

    gt_root  = Path(gt_root)
    pred_dir = Path(pred_dir)

    accs, names = [], []

    # ابحث في مجلدات train مباشرة
    seq_dirs = []
    for d in sorted(gt_root.iterdir()):
        if d.is_dir() and (d / "gt" / "gt.txt").exists():
            seq_dirs.append(d)

    if not seq_dirs:
        print(f"[EVAL] No GT sequences found in {gt_root}")
        return

    print(f"[EVAL] Found {len(seq_dirs)} sequences with GT")

    for seq_dir in seq_dirs:
        gt_file = seq_dir / "gt" / "gt.txt"
        raw_name = seq_dir.name   # e.g. "MOT17-02-SDP"
        parts    = raw_name.split("-")
        base_id  = parts[1] if len(parts) >= 2 else raw_name

        # ابحث عن ملف prediction
        pred_file = None
        candidates = []
        for det in ["SDP", "FRCNN", "DPM"]:
            seq_name = f"MOT17-{base_id}-{det}"
            candidates += [
                pred_dir / f"{seq_name}.txt",
                pred_dir / "data" / seq_name / f"{seq_name}.txt",
                pred_dir / f"MOT17-{base_id}.txt",
            ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                pred_file = c; break

        if pred_file is None:
            print(f"  [SKIP] No prediction for {raw_name} (looked in {len(candidates)} paths)")
            continue

        try:
            # قراءة GT مع تجاهل الأسطر المُعلَّق عليها
            gt_data = []
            with open(gt_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    vals = line.split(',')
                    if len(vals) < 6: continue
                    gt_data.append([float(v) for v in vals[:10]])
            gt_arr = np.array(gt_data, dtype=np.float64)

            # فلترة: only pedestrians (class=1) that are not ignored (conf=1)
            if gt_arr.shape[1] >= 8:
                mask = (gt_arr[:, 6] == 1) & (gt_arr[:, 7] == 1)
                gt_arr = gt_arr[mask]

            # قراءة predictions
            pred_data = []
            with open(pred_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    vals = line.split(',')
                    if len(vals) < 6: continue
                    pred_data.append([float(v) for v in vals[:10]])
            pred_arr = np.array(pred_data, dtype=np.float64) if pred_data else np.zeros((0,10))

            # بناء DataFrames يدوياً لتجنب np.asfarray
            import pandas as pd

            def build_df(arr):
                if len(arr) == 0:
                    return pd.DataFrame(columns=['FrameId','Id','X','Y','Width','Height',
                                                  'Confidence','ClassId','Visibility','unused'])
                cols = ['FrameId','Id','X','Y','Width','Height','Confidence','ClassId','Visibility','unused']
                n_cols = min(arr.shape[1], len(cols))
                df = pd.DataFrame(arr[:, :n_cols], columns=cols[:n_cols])
                df['FrameId'] = df['FrameId'].astype(int)
                df['Id']      = df['Id'].astype(int)
                return df.set_index(['FrameId','Id'])

            gt_df   = build_df(gt_arr)
            pred_df = build_df(pred_arr)

            acc = mm.MOTAccumulator()
            all_frames = sorted(set(
                list(gt_df.index.get_level_values('FrameId').unique()) +
                list(pred_df.index.get_level_values('FrameId').unique())
            ))

            for fid in all_frames:
                gt_ids, pred_ids = [], []
                gt_boxes, pred_boxes = [], []

                if fid in gt_df.index.get_level_values('FrameId'):
                    gf = gt_df.loc[fid]
                    if isinstance(gf, pd.Series): gf = gf.to_frame().T
                    gt_ids   = list(gf.index)
                    gt_boxes = gf[['X','Y','Width','Height']].values.tolist()

                if fid in pred_df.index.get_level_values('FrameId'):
                    pf = pred_df.loc[fid]
                    if isinstance(pf, pd.Series): pf = pf.to_frame().T
                    pred_ids   = list(pf.index)
                    pred_boxes = pf[['X','Y','Width','Height']].values.tolist()

                # حساب IoU distances
                if gt_boxes and pred_boxes:
                    dists = mm.distances.iou_matrix(
                        np.array(gt_boxes, dtype=np.float64),
                        np.array(pred_boxes, dtype=np.float64),
                        max_iou=0.5
                    )
                else:
                    dists = np.empty((len(gt_ids), len(pred_ids)))
                    dists[:] = np.nan

                acc.update(gt_ids, pred_ids, dists, frameid=fid)

            accs.append(acc)
            label = f"MOT17-{base_id}"
            if label not in names:
                names.append(label)
            else:
                names.append(f"MOT17-{base_id}-{raw_name.split('-')[-1]}")
            print(f"  [OK] {raw_name} → {pred_file.name}")

        except Exception as e:
            print(f"  [WARN] {raw_name}: {e}")
            import traceback; traceback.print_exc()

    if not accs:
        print("[EVAL] No sequences evaluated."); return

    # ── حساب المقاييس ────────────────────────────────────────────────────────
    mh      = mm.metrics.create()
    metrics = ["num_frames","mota","motp","idf1",
               "num_switches","num_misses","num_false_positives",
               "num_detections","mostly_tracked","mostly_lost","num_fragmentations"]

    # دمج نتائج نفس التسلسل (DPM+FRCNN+SDP → نتيجة واحدة)
    unique_ids = sorted(set(n.replace("MOT17-","").split("-")[0] for n in names))
    merged_accs, merged_names = [], []
    for uid in unique_ids:
        idxs = [i for i,n in enumerate(names) if f"-{uid}" in n or n.endswith(uid)]
        if not idxs: continue
        if len(idxs) == 1:
            merged_accs.append(accs[idxs[0]])
        else:
            # دمج الـ accumulators
            merged = mm.MOTAccumulator.merge_event_dataframes(
                [accs[i].events for i in idxs]
            )
            ma = mm.MOTAccumulator()
            ma._events = merged
            merged_accs.append(ma)
        merged_names.append(f"MOT17-{uid}")

    summary = mh.compute_many(
        merged_accs, names=merged_names,
        metrics=metrics,
        generate_overall=True
    )

    # ── رسم الجدول ───────────────────────────────────────────────────────────
    def color(val, good, warn):
        if val >= good:  return "🟢"
        if val >= warn:  return "🟡"
        return "🔴"

    rows = []
    for name in merged_names + ["OVERALL"]:
        r    = summary.loc[name]
        mota = float(r["mota"])  * 100
        motp = float(r["motp"])  * 100
        idf1 = float(r["idf1"])  * 100
        idsw = int(r["num_switches"])
        fn   = int(r["num_misses"])
        fp   = int(r["num_false_positives"])
        tp   = int(r["num_detections"])
        mt   = int(r["mostly_tracked"])
        ml   = int(r["mostly_lost"])
        frag = int(r["num_fragmentations"])
        rows.append((name, mota, motp, idf1, idsw, fn, fp, tp, mt, ml, frag))

    COL_W   = [12, 8, 8, 8, 6, 8, 8, 8, 5, 5, 6]
    HEADERS = ["Sequence","MOTA%","MOTP%","IDF1%","IDSW","FN","FP","TP","MT","ML","Frag"]
    sep     = "+" + "+".join("-"*(w+2) for w in COL_W) + "+"
    def row_str(cells):
        return "|" + "|".join(f" {str(c):<{w}} " for c,w in zip(cells, COL_W)) + "|"

    total_w = sum(COL_W) + len(COL_W)*3 - 1
    print("\n")
    print("╔" + "═"*total_w + "╗")
    print("║" + "📊  MOT17 EVALUATION RESULTS".center(total_w) + "║")
    print("╚" + "═"*total_w + "╝")
    print(sep)
    print(row_str(HEADERS))
    print(sep.replace("-","="))

    for name, mota, motp, idf1, idsw, fn, fp, tp, mt, ml, frag in rows:
        is_overall = (name == "OVERALL")
        cells = [
            f"{'★ ' if is_overall else ''}{name}",
            f"{color(mota,70,60)}{mota:5.1f}",
            f"{motp:6.1f}",
            f"{color(idf1,70,60)}{idf1:5.1f}",
            f"{idsw:5d}",
            f"{fn:7d}",
            f"{fp:7d}",
            f"{tp:7d}",
            f"{mt:4d}",
            f"{ml:4d}",
            f"{frag:5d}",
        ]
        if is_overall: print(sep.replace("-","="))
        print(row_str(cells))
    print(sep)

    overall = rows[-1]
    _, mota, motp, idf1, idsw, fn, fp, tp, mt, ml, frag = overall
    target_mota = mota >= 70
    target_idf1 = idf1 >= 70

    print(f"""
┌─────────────────────────────────────────┐
│          OVERALL PERFORMANCE            │
├─────────────────────────────────────────┤
│  🎯 MOTA  : {mota:6.2f}%  {'✅ TARGET!' if target_mota else '⬆️  Target: 70%'}{'       ' if target_mota else '   '}│
│  🔗 IDF1  : {idf1:6.2f}%  {'✅ TARGET!' if target_idf1 else '⬆️  Target: 70%'}{'       ' if target_idf1 else '   '}│
│  📍 MOTP  : {motp:6.2f}%                    │
├─────────────────────────────────────────┤
│  🔀 IDSW  : {idsw:>8,}  (Identity Switches) │
│  ❌ FN    : {fn:>8,}  (False Negatives)   │
│  ⚠️  FP    : {fp:>8,}  (False Positives)   │
│  ✅ TP    : {tp:>8,}  (True Positives)    │
├─────────────────────────────────────────┤
│  📈 MT    : {mt:>8,}  (Mostly Tracked)    │
│  📉 ML    : {ml:>8,}  (Mostly Lost)       │
│  💔 Frag  : {frag:>8,}  (Fragmentations)   │
└─────────────────────────────────────────┘""")

    print("\n  💡 IMPROVEMENT HINTS:")
    if fn > fp * 3:
        print("     → FN عالٍ: خفّض YOLO_CONF أو زد INTERPOLATION_MAX_GAP")
    if idsw > len(merged_names) * 80:
        print("     → IDSW عالٍ: فعّل ReID أو زد TRACK_BUFFER")
    if frag > len(merged_names) * 60:
        print("     → Frag عالٍ: طبّق GSI أو GBR postprocessing")
    if mota < 65:
        print("     → MOTA منخفض: جرّب fine-tune YOLO أو YOLOv9e")
    if mota >= 70 and idf1 >= 70:
        print("     ✅ ممتاز! النظام وصل الهدف 70%+")
    print()

    # حفظ CSV
    csv_path = Path(pred_dir) / "evaluation_results.csv"
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Sequence","MOTA","MOTP","IDF1","IDSW","FN","FP","TP","MT","ML","Frag"])
            for r in rows:
                writer.writerow([r[0],f"{r[1]:.2f}",f"{r[2]:.2f}",f"{r[3]:.2f}",
                                  r[4],r[5],r[6],r[7],r[8],r[9],r[10]])
        print(f"  💾 Results saved: {csv_path}")
    except Exception as e:
        print(f"  [WARN] CSV save failed: {e}")



# ==========================================================================================
# MAIN
# ==========================================================================================

def main():
    # ── تثبيت المتطلبات على Kaggle ──────────────────────────────────────────
    import subprocess, sys
    for pkg in ["boxmot", "motmetrics"]:
        try:
            __import__(pkg.replace("-","_"))
        except ImportError:
            print(f"[SETUP] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

    # ── تحديد وضع التشغيل ───────────────────────────────────────────────────
    global RUN_MODE
    if RUN_MODE == "train":
        config.ROOT_DIR = config.KAGGLE_MOT17 / "train"
        active_ids      = MOT17_TRAIN_BASE_IDS
        print("▶ MODE: TRAIN (local evaluation with GT)")
    else:
        config.ROOT_DIR = config.KAGGLE_MOT17 / "test"
        active_ids      = MOT17_TEST_BASE_IDS
        print("▶ MODE: TEST  (submission generation)")

    ensure_dir(config.OUTPUT_DIR)
    for f in Path(config.OUTPUT_DIR).glob("*.txt"):
        try: f.unlink()
        except: pass

    print("="*80)
    print("MOT17 HYBRID FUZZY TRACKING — IMPROVED v2.0 + FINE-TUNE")
    print("="*80)
    print(f"CUDA        : {torch.cuda.is_available()}" +
          (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"ROOT_DIR    : {config.ROOT_DIR}")
    print(f"OUTPUT      : {config.OUTPUT_DIR}")
    print(f"FAST_MODE   : {config.FAST_MODE}  |  SKIP_FRCNN : {config.SKIP_FRCNN}")
    print(f"IMG_SIZE    : {'640 (fast)' if config.FAST_MODE else '1280 (full)'}")
    print(f"MAX_FRAMES  : {config.MAX_FRAMES or 'ALL'}")
    print(f"YOLO_CONF   : {config.YOLO_CONF}  |  FRCNN_CONF : {config.FRCNN_CONF}")
    print(f"FINETUNE    : {config.FINETUNE_ENABLED}  |  EPOCHS: {config.FINETUNE_EPOCHS}")
    print(f"SEQUENCES   : {active_ids}")
    est_finetune = config.FINETUNE_EPOCHS * 2 if config.FINETUNE_ENABLED else 0
    est_tracking = len(active_ids) * (3 if config.SKIP_FRCNN else 15)
    print(f"EST. TIME   : ~{est_finetune + est_tracking} minutes total")
    print("="*80)

    if ByteTrack is None:
        raise ImportError("ByteTrack not available. Run: pip install boxmot")

    # ── STEP 0: Fine-tune YOLO على MOT17 ────────────────────────────────────
    if config.FINETUNE_ENABLED:
        print("\n" + "━"*80)
        print("STEP 0/5 — YOLO FINE-TUNE")
        print("━"*80)
        ft_success = finetune_pipeline()
        if ft_success:
            print_finetune_results()
        print("━"*80 + "\n")

    # ── STEP 1: تحميل النماذج ────────────────────────────────────────────────
    yolo_model  = load_yolo()
    frcnn_model = load_frcnn()
    produced    = []

    for base_id in active_ids:
        seq_dir = find_sequence_folder(config.ROOT_DIR, base_id)
        if seq_dir is None:
            print(f"\n[SKIP] MOT17-{base_id} — not found in {config.ROOT_DIR}")
            continue

        print(f"\n[RUN] MOT17-{base_id}  ←  {seq_dir.name}")
        result = process_sequence(seq_dir, yolo_model, frcnn_model)
        if result is None:
            print(f"[SKIP] MOT17-{base_id} — processing failed"); continue

        _, tracks = result

        if RUN_MODE == "train":
            # في وضع train: احفظ بالهيكل الكامل للتقييم المحلي
            names = save_motchallenge_structure(base_id, tracks, config.OUTPUT_DIR)
            produced.extend(names)
            print(f"[DONE] MOT17-{base_id} → {len(names)} files")
        else:
            # في وضع test: أنتج كل 42 ملف للـ submission
            names = generate_all_42_files(base_id, tracks, config.OUTPUT_DIR)
            produced.extend(names)
            print(f"[DONE] MOT17-{base_id} → {len(names)} files")

    print(f"\n[INFO] Produced {len(produced)} tracking files.")

    # ── تقييم محلي (train mode فقط) ─────────────────────────────────────────
    if RUN_MODE == "train":
        print("\n[EVAL] Running local evaluation...")
        evaluate_results(config.ROOT_DIR, config.OUTPUT_DIR)

    # ── بناء ZIP (test mode فقط) ─────────────────────────────────────────────
    else:
        ensure_all_42_placeholders(config.OUTPUT_DIR)
        build_submission_zip(config.OUTPUT_DIR, config.ZIP_PATH)
        zip_ok = verify_zip(config.ZIP_PATH)

        with zipfile.ZipFile(config.ZIP_PATH, "r") as zf:
            zip_names = [n.replace(".txt","") for n in zf.namelist()]
        missing = [n for n in MOT17_ALL_42 if n not in zip_names]

        print("\n" + "="*80)
        print("SUBMISSION READY")
        print("="*80)
        print(f"ZIP    : {config.ZIP_PATH}")
        print(f"FILES  : {len(zip_names)} / 42")
        print(f"VALID  : {'PASS ✅' if zip_ok else 'FAIL ❌'}")
        print(f"MISSING: {missing if missing else 'None ✓'}")
        print("="*80)


if __name__ == "__main__":
    main()
