import os
import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


# =========================
# 1) 데이터 정의
# =========================

ASSET_DIR = "assets"


@dataclass
class PersonProfile:
    face_shape: str
    eye_type: str
    mood: List[str]
    expression: str
    hair_style: List[str]


def asset_path(filename: str) -> str:
    return os.path.join(ASSET_DIR, filename)


POKEMON_PROFILES: Dict[str, Dict[str, Any]] = {
    "eevee": {
        "display_name": "이브이",
        "emoji": "🦊",
        "image_path": asset_path("eevee.png"),
        "face_shape": ["round", "oval"],
        "eye_type": ["round", "soft"],
        "mood": ["cute", "friendly", "bright"],
        "expression": ["smile", "neutral"],
        "hair_style": ["soft", "fluffy"],
        "stability": 5,
    },
    "pikachu": {
        "display_name": "피카츄",
        "emoji": "⚡",
        "image_path": asset_path("pikachu.png"),
        "face_shape": ["round"],
        "eye_type": ["round"],
        "mood": ["cute", "friendly", "playful", "bright"],
        "expression": ["smile", "energetic"],
        "hair_style": ["simple"],
        "stability": 5,
    },
    "jigglypuff": {
        "display_name": "푸린",
        "emoji": "🎤",
        "image_path": asset_path("jigglypuff.png"),
        "face_shape": ["round"],
        "eye_type": ["round", "soft"],
        "mood": ["cute", "lovely", "friendly"],
        "expression": ["smile", "neutral"],
        "hair_style": ["simple", "soft"],
        "stability": 4,
    },
    "espeon": {
        "display_name": "에브이",
        "emoji": "🔮",
        "image_path": asset_path("espeon.png"),
        "face_shape": ["oval", "sharp"],
        "eye_type": ["sharp", "dreamy"],
        "mood": ["calm", "chic", "mysterious"],
        "expression": ["neutral", "reserved"],
        "hair_style": ["sleek", "sharp"],
        "stability": 4,
    },
    "umbreon": {
        "display_name": "블래키",
        "emoji": "🌙",
        "image_path": asset_path("umbreon.png"),
        "face_shape": ["sharp", "oval"],
        "eye_type": ["sharp"],
        "mood": ["chic", "dark", "mysterious"],
        "expression": ["neutral", "reserved"],
        "hair_style": ["sleek", "sharp"],
        "stability": 4,
    },
    "ralts": {
        "display_name": "랄토스",
        "emoji": "🍃",
        "image_path": asset_path("ralts.png"),
        "face_shape": ["oval", "long"],
        "eye_type": ["soft", "dreamy"],
        "mood": ["calm", "mysterious", "gentle"],
        "expression": ["neutral", "reserved"],
        "hair_style": ["bangs", "soft"],
        "stability": 4,
    },
    "gardevoir": {
        "display_name": "가디안",
        "emoji": "✨",
        "image_path": asset_path("gardevoir.png"),
        "face_shape": ["long", "oval"],
        "eye_type": ["soft", "dreamy"],
        "mood": ["elegant", "mysterious", "calm"],
        "expression": ["neutral", "reserved"],
        "hair_style": ["sleek", "flowing"],
        "stability": 4,
    },
    "charmander": {
        "display_name": "파이리",
        "emoji": "🔥",
        "image_path": asset_path("charmander.png"),
        "face_shape": ["round"],
        "eye_type": ["round", "sharp"],
        "mood": ["playful", "bright", "energetic"],
        "expression": ["smile", "energetic"],
        "hair_style": ["simple"],
        "stability": 4,
    },
    "torchic": {
        "display_name": "아차모",
        "emoji": "🐥",
        "image_path": asset_path("torchic.png"),
        "face_shape": ["round"],
        "eye_type": ["round"],
        "mood": ["cute", "playful", "bright"],
        "expression": ["smile", "energetic"],
        "hair_style": ["fluffy", "simple"],
        "stability": 4,
    },
}

WEIGHTS = {
    "face_shape": 25,
    "eye_type": 20,
    "mood": 20,
    "hair_style": 10,
    "expression": 15,
    "stability": 10,
}

SIMILAR_MAP = {
    "face_shape": {
        "round": ["oval"],
        "oval": ["round", "long", "sharp"],
        "long": ["oval", "sharp"],
        "sharp": ["oval", "long"],
    },
    "eye_type": {
        "round": ["soft"],
        "soft": ["round", "dreamy"],
        "sharp": ["dreamy"],
        "dreamy": ["soft", "sharp"],
    },
    "expression": {
        "smile": ["energetic", "neutral"],
        "neutral": ["reserved", "smile"],
        "energetic": ["smile"],
        "reserved": ["neutral"],
    },
}


# =========================
# 2) MediaPipe 유틸
# =========================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_landmark_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def estimate_face_shape(landmarks, w, h) -> str:
    left_cheek = get_landmark_point(landmarks, 234, w, h)
    right_cheek = get_landmark_point(landmarks, 454, w, h)
    forehead = get_landmark_point(landmarks, 10, w, h)
    chin = get_landmark_point(landmarks, 152, w, h)

    face_width = euclidean(left_cheek, right_cheek)
    face_height = euclidean(forehead, chin)
    ratio = face_height / max(face_width, 1e-6)

    if ratio < 1.15:
        return "round"
    elif ratio < 1.35:
        return "oval"
    elif ratio < 1.50:
        return "sharp"
    return "long"


def estimate_eye_type(landmarks, w, h) -> str:
    left_eye_outer = get_landmark_point(landmarks, 33, w, h)
    left_eye_inner = get_landmark_point(landmarks, 133, w, h)
    left_eye_top = get_landmark_point(landmarks, 159, w, h)
    left_eye_bottom = get_landmark_point(landmarks, 145, w, h)

    eye_width = euclidean(left_eye_outer, left_eye_inner)
    eye_height = euclidean(left_eye_top, left_eye_bottom)
    ratio = eye_height / max(eye_width, 1e-6)
    slope = (left_eye_outer[1] - left_eye_inner[1]) / max(abs(left_eye_outer[0] - left_eye_inner[0]), 1e-6)

    if ratio > 0.33:
        return "round"
    elif slope < -0.08:
        return "sharp"
    elif ratio > 0.24:
        return "soft"
    return "dreamy"


def estimate_expression(landmarks, w, h) -> str:
    left_mouth = get_landmark_point(landmarks, 61, w, h)
    right_mouth = get_landmark_point(landmarks, 291, w, h)
    top_lip = get_landmark_point(landmarks, 13, w, h)
    bottom_lip = get_landmark_point(landmarks, 14, w, h)

    mouth_width = euclidean(left_mouth, right_mouth)
    mouth_open = euclidean(top_lip, bottom_lip)

    mouth_corner_avg_y = (left_mouth[1] + right_mouth[1]) / 2
    lip_center_y = (top_lip[1] + bottom_lip[1]) / 2
    smile_signal = lip_center_y - mouth_corner_avg_y

    if smile_signal > 3:
        return "smile"
    elif mouth_open / max(mouth_width, 1e-6) > 0.30:
        return "energetic"
    elif smile_signal > 1:
        return "neutral"
    return "reserved"


def estimate_mood(face_shape: str, eye_type: str, expression: str) -> List[str]:
    mood = []

    if face_shape == "round":
        mood.append("cute")
    if face_shape in ["oval", "sharp"]:
        mood.append("chic")
    if face_shape == "long":
        mood.append("elegant")

    if eye_type == "round":
        mood.append("friendly")
    elif eye_type == "soft":
        mood.append("calm")
    elif eye_type in ["sharp", "dreamy"]:
        mood.append("mysterious")

    if expression == "smile":
        mood.append("bright")
    elif expression == "energetic":
        mood.append("playful")
    else:
        mood.append("calm")

    return list(sorted(set(mood)))


def estimate_hair_style_stub(face_shape: str, eye_type: str) -> List[str]:
    if face_shape == "round":
        return ["soft"]
    if eye_type == "sharp":
        return ["sleek"]
    return ["soft"]


def extract_person_profile_from_image(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_bgr.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            raise ValueError("얼굴을 찾지 못했습니다. 정면 사진이나 얼굴이 크게 나온 사진으로 다시 시도해 주세요.")

        face_landmarks = result.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        face_shape = estimate_face_shape(landmarks, w, h)
        eye_type = estimate_eye_type(landmarks, w, h)
        expression = estimate_expression(landmarks, w, h)
        mood = estimate_mood(face_shape, eye_type, expression)
        hair_style = estimate_hair_style_stub(face_shape, eye_type)

        annotated = image_bgr.copy()
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

        person = PersonProfile(
            face_shape=face_shape,
            eye_type=eye_type,
            mood=mood,
            expression=expression,
            hair_style=hair_style,
        )
        return person, annotated


# =========================
# 3) 매칭 로직
# =========================

def score_single_label(person_value: str, pokemon_values: List[str], category: str) -> int:
    if person_value in pokemon_values:
        return 5
    similar = SIMILAR_MAP.get(category, {}).get(person_value, [])
    if any(v in pokemon_values for v in similar):
        return 3
    return 1


def score_multi_label(person_values: List[str], pokemon_values: List[str]) -> int:
    overlap = len(set(person_values) & set(pokemon_values))
    if overlap >= 2:
        return 5
    elif overlap == 1:
        return 3
    return 1


def compute_total(raw_scores: Dict[str, int], stability: int) -> Tuple[int, Dict[str, float]]:
    breakdown = {
        "face_shape": raw_scores["face_shape"] / 5 * WEIGHTS["face_shape"],
        "eye_type": raw_scores["eye_type"] / 5 * WEIGHTS["eye_type"],
        "mood": raw_scores["mood"] / 5 * WEIGHTS["mood"],
        "hair_style": raw_scores["hair_style"] / 5 * WEIGHTS["hair_style"],
        "expression": raw_scores["expression"] / 5 * WEIGHTS["expression"],
        "stability": stability / 5 * WEIGHTS["stability"],
    }
    total = round(sum(breakdown.values()))
    return total, breakdown


def build_explanation(person: PersonProfile, pokemon: Dict[str, Any], total: int) -> str:
    reasons = []
    if person.face_shape in pokemon["face_shape"]:
        reasons.append("얼굴형이 잘 맞음")
    if person.eye_type in pokemon["eye_type"]:
        reasons.append("눈매 유사도가 높음")
    if set(person.mood) & set(pokemon["mood"]):
        reasons.append("분위기 톤이 비슷함")
    if person.expression in pokemon["expression"]:
        reasons.append("표정 호환성이 좋음")
    if set(person.hair_style) & set(pokemon["hair_style"]):
        reasons.append("실루엣 연결이 자연스러움")

    if not reasons:
        reasons.append("전체 밸런스가 크게 어색하지 않음")

    return f"{pokemon['display_name']}은(는) {', '.join(reasons[:3])} → 총 {total}점"


def recommend_pokemon(person: PersonProfile, top_k: int = 3):
    results = []

    for _, pokemon in POKEMON_PROFILES.items():
        raw_scores = {
            "face_shape": score_single_label(person.face_shape, pokemon["face_shape"], "face_shape"),
            "eye_type": score_single_label(person.eye_type, pokemon["eye_type"], "eye_type"),
            "mood": score_multi_label(person.mood, pokemon["mood"]),
            "hair_style": score_multi_label(person.hair_style, pokemon["hair_style"]),
            "expression": score_single_label(person.expression, pokemon["expression"], "expression"),
        }

        total, breakdown = compute_total(raw_scores, pokemon["stability"])
        results.append({
            "name": pokemon["display_name"],
            "emoji": pokemon.get("emoji", "⭐"),
            "image_path": pokemon.get("image_path", ""),
            "score": total,
            "explanation": build_explanation(person, pokemon, total),
            "breakdown": breakdown,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =========================
# 4) 표시용 함수
# =========================

def translate_eye_type(value: str) -> str:
    return {
        "sharp": "날카로운 눈매",
        "round": "동그란 눈매",
        "soft": "부드러운 눈매",
        "dreamy": "몽환적인 눈매",
    }.get(value, value)


def translate_expression(value: str) -> str:
    return {
        "smile": "미소형",
        "neutral": "중립형",
        "energetic": "활발형",
        "reserved": "차분형",
    }.get(value, value)


def translate_face_shape(value: str) -> str:
    return {
        "round": "둥근형",
        "oval": "계란형",
        "sharp": "갸름/각진형",
        "long": "긴 얼굴형",
    }.get(value, value)


def translate_moods(values: List[str]) -> str:
    mapping = {
        "cute": "귀여움",
        "friendly": "친근함",
        "bright": "밝음",
        "playful": "장난기",
        "calm": "차분함",
        "chic": "세련됨",
        "mysterious": "신비로움",
        "elegant": "우아함",
    }
    return ", ".join(mapping.get(v, v) for v in values)


def make_feature_html(person: PersonProfile) -> str:
    return f"""
    <div class="feature-card">
        <h3>추출된 얼굴 특징</h3>
        <div class="feature-grid">
            <div class="mini-box"><span>얼굴형</span><strong>{translate_face_shape(person.face_shape)}</strong></div>
            <div class="mini-box"><span>눈매</span><strong>{translate_eye_type(person.eye_type)}</strong></div>
            <div class="mini-box"><span>표정</span><strong>{translate_expression(person.expression)}</strong></div>
            <div class="mini-box"><span>분위기</span><strong>{translate_moods(person.mood)}</strong></div>
        </div>
    </div>
    """


def score_badge(score: int) -> str:
    if score >= 90:
        label = "매우 적합"
        cls = "badge excellent"
    elif score >= 80:
        label = "적합"
        cls = "badge good"
    elif score >= 70:
        label = "조건부 적합"
        cls = "badge fair"
    else:
        label = "낮음"
        cls = "badge low"
    return f'<span class="{cls}">{score}점 · {label}</span>'


def img_tag_or_fallback(result: Dict[str, Any], class_name: str) -> str:
    path = result.get("image_path", "")
    if path and os.path.exists(path):
        return f'<img src="/file={path}" class="{class_name}" alt="{result["name"]}">'
    return f'<div class="{class_name} fallback">{result.get("emoji","⭐")}</div>'


def make_top1_html(result: Dict[str, Any]) -> str:
    b = result["breakdown"]
    return f"""
    <div class="top-card">
        <div class="top-head">
            {img_tag_or_fallback(result, "top-thumb")}
            <div>
                <div class="top-label">가장 잘 어울리는 포켓몬</div>
                <h2>{result["name"]}</h2>
                {score_badge(result["score"])}
            </div>
        </div>
        <p class="top-desc">{result["explanation"]}</p>
        <div class="bar-group">
            <div class="bar-row"><span>얼굴형</span><progress value="{b['face_shape']}" max="25"></progress><strong>{b['face_shape']:.1f}/25</strong></div>
            <div class="bar-row"><span>눈매</span><progress value="{b['eye_type']}" max="20"></progress><strong>{b['eye_type']:.1f}/20</strong></div>
            <div class="bar-row"><span>분위기</span><progress value="{b['mood']}" max="20"></progress><strong>{b['mood']:.1f}/20</strong></div>
            <div class="bar-row"><span>실루엣</span><progress value="{b['hair_style']}" max="10"></progress><strong>{b['hair_style']:.1f}/10</strong></div>
            <div class="bar-row"><span>표정</span><progress value="{b['expression']}" max="15"></progress><strong>{b['expression']:.1f}/15</strong></div>
            <div class="bar-row"><span>안정성</span><progress value="{b['stability']}" max="10"></progress><strong>{b['stability']:.1f}/10</strong></div>
        </div>
    </div>
    """


def make_rank_card(result: Dict[str, Any], rank: int) -> str:
    b = result["breakdown"]
    return f"""
    <div class="rank-card">
        <div class="rank-title">
            <div class="rank-left">
                <span class="rank-num">#{rank}</span>
                {img_tag_or_fallback(result, "rank-thumb")}
                <span class="rank-name">{result["name"]}</span>
            </div>
            {score_badge(result["score"])}
        </div>
        <p>{result["explanation"]}</p>
        <div class="mini-score-grid">
            <div>얼굴형 <strong>{b['face_shape']:.1f}</strong></div>
            <div>눈매 <strong>{b['eye_type']:.1f}</strong></div>
            <div>분위기 <strong>{b['mood']:.1f}</strong></div>
            <div>실루엣 <strong>{b['hair_style']:.1f}</strong></div>
            <div>표정 <strong>{b['expression']:.1f}</strong></div>
            <div>안정성 <strong>{b['stability']:.1f}</strong></div>
        </div>
    </div>
    """


def make_all_rank_html(results: List[Dict[str, Any]]) -> str:
    cards = [make_rank_card(r, i + 1) for i, r in enumerate(results)]
    return '<div class="rank-wrap">' + "".join(cards) + "</div>"


# =========================
# 5) 예측 함수
# =========================

def predict(image: np.ndarray):
    if image is None:
        empty = "<div class='empty-box'>이미지를 업로드해 주세요.</div>"
        return None, empty, empty, empty

    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        person, annotated_bgr = extract_person_profile_from_image(image_bgr)
        results = recommend_pokemon(person, top_k=3)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        feature_html = make_feature_html(person)
        top1_html = make_top1_html(results[0])
        all_rank_html = make_all_rank_html(results)

        return annotated_rgb, feature_html, top1_html, all_rank_html

    except Exception as e:
        err = f"<div class='empty-box'>오류: {str(e)}</div>"
        return None, err, err, err


# =========================
# 6) UI
# =========================

CUSTOM_CSS = """
.gradio-container { max-width: 1180px !important; }
.hero {
    padding: 18px 22px;
    border-radius: 20px;
    background: linear-gradient(135deg, #fff7ed, #fef3c7);
    border: 1px solid #fde68a;
    margin-bottom: 14px;
}
.hero h1 { margin: 0 0 8px 0; font-size: 30px; }
.hero p { margin: 0; color: #5b4a2f; line-height: 1.6; }

.feature-card, .top-card, .rank-card, .empty-box {
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    background: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}
.feature-card { padding: 18px; }
.feature-card h3 { margin-top: 0; margin-bottom: 12px; }
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.mini-box {
    padding: 12px;
    border-radius: 14px;
    background: #f9fafb;
    border: 1px solid #eef2f7;
}
.mini-box span {
    display: block;
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 4px;
}
.mini-box strong { font-size: 15px; }

.top-card { padding: 22px; }
.top-head {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 10px;
}
.top-label {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 4px;
}
.top-head h2 {
    margin: 0 0 8px 0;
    font-size: 28px;
}
.top-desc {
    margin-top: 8px;
    margin-bottom: 16px;
    line-height: 1.7;
}

.top-thumb, .rank-thumb {
    object-fit: contain;
    background: #fff7ed;
    border: 1px solid #fed7aa;
}
.top-thumb {
    width: 88px;
    height: 88px;
    border-radius: 22px;
    padding: 8px;
}
.rank-thumb {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    padding: 4px;
}
.fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
}
.top-thumb.fallback { width: 88px; height: 88px; }
.rank-thumb.fallback { width: 48px; height: 48px; font-size: 20px; }

.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}
.badge.excellent { background: #dcfce7; color: #166534; }
.badge.good { background: #dbeafe; color: #1d4ed8; }
.badge.fair { background: #fef3c7; color: #92400e; }
.badge.low { background: #fee2e2; color: #b91c1c; }

.bar-group { display: grid; gap: 10px; }
.bar-row {
    display: grid;
    grid-template-columns: 70px 1fr 70px;
    align-items: center;
    gap: 10px;
    font-size: 14px;
}
.bar-row progress {
    width: 100%;
    height: 12px;
}

.rank-wrap { display: grid; gap: 14px; }
.rank-card { padding: 18px; }
.rank-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
}
.rank-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.rank-num {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    background: #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}
.rank-name {
    font-size: 20px;
    font-weight: 700;
}
.rank-card p {
    margin-top: 12px;
    line-height: 1.7;
}
.mini-score-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 12px;
}
.mini-score-grid > div {
    background: #f9fafb;
    border: 1px solid #eef2f7;
    border-radius: 12px;
    padding: 10px;
    font-size: 14px;
}
.empty-box {
    padding: 18px;
    color: #6b7280;
}
footer { display:none !important; }
"""


with gr.Blocks(css=CUSTOM_CSS, title="사람 얼굴 ↔ 포켓몬 매칭 데모") as demo:
    gr.HTML("""
    <div class="hero">
        <h1>사람 얼굴 ↔ 포켓몬 매칭 데모</h1>
        <p>
            얼굴 사진을 업로드하면 얼굴형, 눈매, 표정, 분위기를 추정한 뒤
            가장 잘 어울리는 포켓몬 Top 3를 썸네일 카드 형태로 보여줍니다.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="얼굴 사진 업로드")
            run_btn = gr.Button("추천 결과 보기", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="얼굴 랜드마크 시각화")

    with gr.Row():
        feature_html = gr.HTML("<div class='empty-box'>아직 분석 전입니다.</div>")
        top1_html = gr.HTML("<div class='empty-box'>가장 잘 어울리는 포켓몬이 여기에 표시됩니다.</div>")

    rank_html = gr.HTML("<div class='empty-box'>Top 3 결과 카드가 여기에 표시됩니다.</div>")

    run_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, feature_html, top1_html, rank_html]
    )

demo.launch(allowed_paths=[ASSET_DIR])