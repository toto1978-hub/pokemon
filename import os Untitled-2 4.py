import os
import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import base64
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Python 3.13 + mediapipe 0.10.x 환경에서는 mp.solutions가 비어 있을 수 있어
# OpenCV 기반 폴백 분석기로 동작하도록 분기한다.
MP_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


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
        "types": ["노말"],
        "keywords": ["귀여움", "친근함", "부드러움"],
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
        "types": ["전기"],
        "keywords": ["밝음", "활발함", "대중성"],
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
        "types": ["노말", "페어리"],
        "keywords": ["사랑스러움", "둥근형", "귀여움"],
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
        "types": ["에스퍼"],
        "keywords": ["세련됨", "시크함", "신비로움"],
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
        "types": ["악"],
        "keywords": ["시크함", "어두운 매력", "개성"],
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
        "types": ["에스퍼", "페어리"],
        "keywords": ["몽환적", "조용함", "신비로움"],
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
        "types": ["에스퍼", "페어리"],
        "keywords": ["우아함", "신비로움", "성숙함"],
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
        "types": ["불꽃"],
        "keywords": ["장난기", "활발함", "생동감"],
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
        "types": ["불꽃"],
        "keywords": ["발랄함", "귀여움", "친근함"],
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

TYPE_CLASS_MAP = {
    "노말": "type-normal",
    "전기": "type-electric",
    "불꽃": "type-fire",
    "에스퍼": "type-psychic",
    "페어리": "type-fairy",
    "악": "type-dark",
}


if MP_SOLUTIONS_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
else:
    mp_face_mesh = None
    mp_drawing = None
    mp_drawing_styles = None


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

    if MP_SOLUTIONS_AVAILABLE:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3
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

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smiles = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # 증명사진 등 까다로운 환경을 위해 OpenCV 인식 조건을 더 관대하게 하향조정
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
    if len(faces) == 0:
        raise ValueError("얼굴을 찾지 못했습니다. 정면 사진이나 얼굴이 크게 나온 사진으로 다시 시도해 주세요.")

    x, y, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    roi_gray = gray[y:y + fh, x:x + fw]

    ratio = fh / max(fw, 1e-6)
    if ratio < 1.15:
        face_shape = "round"
    elif ratio < 1.35:
        face_shape = "oval"
    elif ratio < 1.50:
        face_shape = "sharp"
    else:
        face_shape = "long"

    detected_eyes = eyes.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18))
    if len(detected_eyes) >= 2:
        eye_type = "round"
    elif len(detected_eyes) == 1:
        eye_type = "soft"
    elif ratio >= 1.45:
        eye_type = "sharp"
    else:
        eye_type = "dreamy"

    detected_smiles = smiles.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(20, 20))
    if len(detected_smiles) > 0:
        expression = "smile"
    elif ratio < 1.2:
        expression = "energetic"
    elif ratio < 1.4:
        expression = "neutral"
    else:
        expression = "reserved"

    mood = estimate_mood(face_shape, eye_type, expression)
    hair_style = estimate_hair_style_stub(face_shape, eye_type)

    annotated = image_bgr.copy()
    cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 200, 255), 2)
    cv2.putText(
        annotated,
        "Fallback detector (OpenCV)",
        (x, max(y - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )

    person = PersonProfile(
        face_shape=face_shape,
        eye_type=eye_type,
        mood=mood,
        expression=expression,
        hair_style=hair_style,
    )
    return person, annotated


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
            "types": pokemon.get("types", []),
            "keywords": pokemon.get("keywords", []),
            "score": total,
            "explanation": build_explanation(person, pokemon, total),
            "breakdown": breakdown,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


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


def img_tag_or_fallback(result: Dict[str, Any], class_name: str) -> str:
    path = result.get("image_path", "")
    if path and os.path.exists(path):
        # Gradio 6.x allowed_paths (assets/) 기반 서빙:
        # 절대경로 대신 파일의 상대경로(assets/filename.png)를 직접 사용
        filename = os.path.basename(path)
        return f'<img src="/file=assets/{filename}" class="{class_name}" alt="{result["name"]}">'
    return f'<div class="{class_name} fallback">{result.get("emoji","⭐")}</div>'


def render_type_badges(types: List[str]) -> str:
    badges = []
    for t in types:
        cls = TYPE_CLASS_MAP.get(t, "type-default")
        badges.append(f'<span class="type-badge {cls}">{t}</span>')
    return "".join(badges)


def render_keyword_badges(keywords: List[str]) -> str:
    return "".join([f'<span class="keyword-badge">{kw}</span>' for kw in keywords])


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


def ring_svg(score: int) -> str:
    radius = 54
    circumference = 2 * 3.1415926535 * radius
    progress = max(0, min(score, 100))
    offset = circumference * (1 - progress / 100)

    return f"""
    <svg class="score-ring" viewBox="0 0 140 140" width="140" height="140">
        <circle cx="70" cy="70" r="{radius}" class="ring-bg"></circle>
        <circle cx="70" cy="70" r="{radius}" class="ring-progress" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"></circle>
        <text x="70" y="66" text-anchor="middle" class="ring-score">{score}</text>
        <text x="70" y="88" text-anchor="middle" class="ring-label">MATCH</text>
    </svg>
    """


def make_feature_html(person: PersonProfile) -> str:
    flavor_texts = [
        "오박사: 흠... 아주 흥미로운 얼굴 형태를 가졌구나! 🔍",
        "앗! 야생의 포켓몬 트레이너가 나타났다! 🎒",
        "지우: 넌 내 포켓몬으로 정했다! ⚡",
        "로켓단: 우리가 누구냐고 물으신다면 대답해 드리는 게 인지상정! 🌹",
        "도감: 삐리릭- 새로운 특징이 데이터에 추가되었습니다. 📱",
        "피카~ 삐까츄! (너랑 닮은 포켓몬을 찾을게!) ⚡",
        "간호순: 당신과 닮은 포켓몬이 완벽하게 회복되었습니다! 🏥",
        "오박사: 이 데이터를 연구소로 가져가 분석해보마! 🧪",
        "무인편: 포켓몬스터, 내가 제일 잘나가! 🎶",
        "웅이: 나의 실눈은 모든 것을 꿰뚫어보고 있지... 😌",
        "단데기: (사용 중인 기술: 단단해지기!) 🛡️",
        "야돈: .......... (생각 중인 것 같다) 💤",
        "레드: ............! (승부를 걸어왔다!) ❗",
        "로이: 난 로켓단의 로이라고 한다! 🥀",
        "이슬이: 내 포켓몬들은 모두 최고라고! 💦",
        "상록숲: 어딘가에서 피카츄의 울음소리가 들린다... ✨",
        "체육관 관장: 자, 너의 실력을 보여줘! 🏟️",
        "포켓몬 센터: 자, 기운 내렴! 💖",
        "포켓몬 도감: 미지의 포켓몬을 발견했을 때의 설렘! 🌟",
        "세레나: 난 포켓몬 퍼포머가 될 거야! 🎀",
        "릴리에: 어머... 정말 귀여운 특징이에요! 🤍"
    ]
    random_text = random.choice(flavor_texts)

    return f"""
    <div class="feature-card">
        <h3>추출된 얼굴 특징</h3>
        <div class="feature-grid">
            <div class="mini-box"><span>얼굴형</span><strong>{translate_face_shape(person.face_shape)}</strong></div>
            <div class="mini-box"><span>눈매</span><strong>{translate_eye_type(person.eye_type)}</strong></div>
            <div class="mini-box"><span>표정</span><strong>{translate_expression(person.expression)}</strong></div>
            <div class="mini-box"><span>분위기</span><strong>{translate_moods(person.mood)}</strong></div>
        </div>
        <div class="flavor-text-box">
            <p>{random_text}</p>
        </div>
    </div>
    """


def make_matching_card(result: Dict[str, Any], rank: int, face_bgr: np.ndarray) -> str:
    b = result["breakdown"]
    
    # 내 얼굴 이미지를 base64로 인코딩하여 직접 삽입
    _, buffer = cv2.imencode('.jpg', face_bgr)
    face_b64 = base64.b64encode(buffer).decode('utf-8')
    face_img_src = f"data:image/jpeg;base64,{face_b64}"

    return f"""
    <div class="matching-card rank-{rank}">
        <div class="card-header">
            <span class="rank-badge">#{rank}</span>
            {score_badge(result["score"])}
        </div>
        
        <div class="card-images">
            <div class="image-box">
                <div class="image-label">내 얼굴</div>
                <img src="{face_img_src}" class="b64-thumb" alt="내 얼굴">
            </div>
            <div class="image-box">
                <div class="image-label">어울리는 포켓몬</div>
                {img_tag_or_fallback(result, "poke-thumb")}
            </div>
        </div>

        <div class="card-info">
            <h2 class="pokemon-name">{result["name"]}</h2>
            <div class="type-row">{render_type_badges(result["types"])}</div>
            <div class="keyword-row compact">{render_keyword_badges(result["keywords"])}</div>
            <p class="explanation-text">{result["explanation"]}</p>
        </div>

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


def predict(image: np.ndarray):
    if image is None:
        empty = "<div class='empty-box'>이미지를 업로드해 주세요.</div>"
        return empty, empty, empty, empty

    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        person, annotated_bgr = extract_person_profile_from_image(image_bgr)
        results = recommend_pokemon(person, top_k=3)

        feature_html = make_feature_html(person)
        
        card1 = make_matching_card(results[0], 1, annotated_bgr)
        card2 = make_matching_card(results[1], 2, annotated_bgr)
        card3 = make_matching_card(results[2], 3, annotated_bgr)

        return feature_html, card1, card2, card3

    except Exception as e:
        err = f"<div class='empty-box'>오류: {str(e)}</div>"
        return err, err, err, err


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');

body, .gradio-container {
    font-family: 'Jua', sans-serif !important;
    background-color: #fffaf0 !important;
}
.gradio-container { max-width: 1240px !important; }

.hero-banner {
    padding: 24px 28px;
    border-radius: 30px;
    background: linear-gradient(135deg, #fef08a, #fde047);
    border: 3px solid #facc15;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px rgba(250, 204, 21, 0.2);
    text-align: center;
}
.hero-banner h1 { margin: 0 0 10px 0; font-size: 36px; color: #713f12; text-shadow: 1px 1px 0px #fff; }
.hero-banner p { margin: 0; color: #854d0e; font-size: 18px; line-height: 1.6; }

.panel, .feature-card, .matching-card, .empty-box {
    border-radius: 24px !important;
    border: 2px solid #fde68a !important;
    background: white !important;
    box-shadow: 0 8px 24px rgba(251, 191, 36, 0.15) !important;
}

.feature-card {
    padding: 22px;
}

.feature-card h3 {
    margin: 0 0 16px 0;
    color: #b45309;
    font-size: 22px;
    text-align: center;
}

.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.mini-box {
    padding: 14px;
    border-radius: 18px;
    background: #fdf4ff;
    border: 2px solid #fbcfe8;
    text-align: center;
}
.mini-box span {
    display: block;
    font-size: 14px;
    color: #db2777;
    margin-bottom: 6px;
}
.mini-box strong { font-size: 18px; color: #831843; }

.flavor-text-box {
    margin-top: 18px;
    padding: 12px;
    background: #fef9c3;
    border: 2px dashed #fde047;
    border-radius: 14px;
    text-align: center;
}
.flavor-text-box p {
    margin: 0;
    color: #b45309;
    font-size: 15px;
    font-weight: bold;
}

.hero-match-card {
    text-align: center;
    height: 100%;
    background: linear-gradient(180deg, #ffffff, #fffaf0);
}
.hero-caption {
    color: #6b7280;
    font-size: 13px;
    margin-bottom: 6px;
}
.hero-name {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}
.hero-ring-wrap {
    display: flex;
    justify-content: center;
    margin: 8px 0 10px 0;
}
.score-ring .ring-bg {
    fill: none;
    stroke: #eceff3;
    stroke-width: 10;
}
.score-ring .ring-progress {
    fill: none;
    stroke: #f59e0b;
    stroke-width: 10;
    stroke-linecap: round;
    transform: rotate(-90deg);
    transform-origin: 70px 70px;
}
.score-ring .ring-score {
    font-size: 28px;
    font-weight: 800;
    fill: #111827;
}
.score-ring .ring-label {
    font-size: 11px;
    font-weight: 700;
    fill: #6b7280;
    letter-spacing: 1px;
}
.hero-badge-row, .hero-type-row, .hero-keyword-row,
.type-row, .keyword-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.hero-badge-row, .hero-type-row, .type-row {
    justify-content: center;
}
.hero-keyword-row {
    justify-content: center;
    margin-top: 4px;
}
.hero-desc {
    margin-top: 14px;
    line-height: 1.7;
    color: #374151;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: normal;
}
.badge.excellent { background: #dcfce7; color: #14532d; border: 2px solid #bbf7d0; }
.badge.good { background: #dbeafe; color: #1e3a8a; border: 2px solid #bfdbfe; }
.badge.fair { background: #fef3c7; color: #78350f; border: 2px solid #fde68a; }
.badge.low { background: #fee2e2; color: #7f1d1d; border: 2px solid #fecaca; }

.type-badge, .keyword-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: normal;
}
.matching-card {
    border-radius: 24px;
    border: 3px solid #fde047;
    background: white;
    box-shadow: 0 10px 25px rgba(250, 204, 21, 0.2);
    padding: 22px;
    display: flex;
    flex-direction: column;
    height: 100%;
}
.matching-card.rank-1 {
    border: 4px solid #fbbf24;
    background: linear-gradient(180deg, #fef9c3, #ffffff);
    transform: scale(1.02);
    box-shadow: 0 15px 35px rgba(251, 191, 36, 0.3);
}
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}
.rank-badge {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    color: #374151;
}
.rank-1 .rank-badge { background: #fbbf24; color: white; }
.card-images {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}
.image-box {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #f9fafb;
    border-radius: 16px;
    padding: 10px;
    border: 1px solid #eef2f7;
}
.image-label {
    font-size: 13px;
    color: #92400e;
    margin-bottom: 8px;
    background: #fef3c7;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid #fde68a;
}
.b64-thumb, .poke-thumb {
    width: 100%;
    aspect-ratio: 1/1;
    object-fit: cover;
    border-radius: 16px;
}
.poke-thumb {
    object-fit: contain;
}
.fallback {
    font-size: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 1/1;
}
.card-info {
    text-align: center;
    margin-bottom: 16px;
    flex-grow: 1;
}
.pokemon-name {
    font-size: 28px;
    color: #b45309;
    margin: 0 0 10px 0;
    text-shadow: 1px 1px 0px #fef3c7;
}
.explanation-text {
    margin-top: 14px;
    font-size: 15px;
    line-height: 1.6;
    color: #78350f;
    background: #fffbeb;
    padding: 10px;
    border-radius: 12px;
}
.mini-score-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: auto;
}
.mini-score-grid > div {
    background: #fef9c3;
    border: 1px solid #fef08a;
    border-radius: 10px;
    padding: 8px;
    font-size: 12px;
    text-align: center;
    color: #854d0e;
}
.mini-score-grid > div strong {
    display: block;
    font-size: 16px;
    color: #713f12;
    margin-top: 2px;
}

.feature-preview-title {
    font-size: 18px;
    color: #b45309;
    text-align: center;
    margin-bottom: 12px;
}
.empty-box {
    padding: 30px;
    color: #b45309;
    font-size: 16px;
    text-align: center;
    background: #fffbeb;
    border-radius: 20px;
    border: 3px dashed #fde047;
}
footer { display:none !important; }
"""

with gr.Blocks(title="사람 얼굴 ↔ 포켓몬 매칭 데모") as demo:
    gr.HTML("""
    <div class="hero-banner">
        <h1>사람 얼굴 ↔ 포켓몬 매칭 데모</h1>
        <p>
            사진 한 장으로 당신과 가장 잘 어울리는 포켓몬 Top 3를 찾아보세요.<br>
            얼굴형, 눈매, 표정, 분위기를 종합적으로 분석합니다.
        </p>
    </div>
    """)

    # 상단 입력 영역 (한 화면에 집중)
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="내 얼굴 사진 업로드 (증명사진, 정면 셀카 등)")
            run_btn = gr.Button("매칭 결과 보기", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.HTML("<div class='feature-preview-title'>분석된 얼굴 특징</div>")
            feature_html = gr.HTML("<div class='empty-box'>사진을 업로드하고 결과를 확인하세요.</div>")

    gr.HTML("<hr style='margin: 30px 0; border: 2px dashed #fde047;'>")
    gr.HTML("<h2 style='text-align: center; margin-bottom: 24px; color: #b45309; font-size: 32px; text-shadow: 1px 1px 0px #fef3c7;'>✨ 매칭 결과 Top 3 ✨</h2>")

    # 하단 결과 영역 (1, 2, 3위 가로 나열)
    with gr.Row():
        with gr.Column(scale=1):
            card1_html = gr.HTML("<div class='empty-box'>1위 결과 대기 중...</div>")
        with gr.Column(scale=1):
            card2_html = gr.HTML("<div class='empty-box'>2위 결과 대기 중...</div>")
        with gr.Column(scale=1):
            card3_html = gr.HTML("<div class='empty-box'>3위 결과 대기 중...</div>")

    run_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[feature_html, card1_html, card2_html, card3_html]
    )

demo.launch(
    allowed_paths=[ASSET_DIR], 
    css=CUSTOM_CSS,
    server_name="127.0.0.1",
    server_port=7865
)