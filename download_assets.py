"""
PokeAPI에서 포켓몬 스프라이트 이미지를 다운로드해 assets/ 폴더에 저장합니다.
"""
import os
import urllib.request

ASSET_DIR = "assets"
os.makedirs(ASSET_DIR, exist_ok=True)

# PokeAPI 공식 스프라이트 (고해상도 artwork)
POKEMON_URLS = {
    "eevee":     "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/133.png",
    "pikachu":   "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
    "jigglypuff":"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/39.png",
    "espeon":    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/196.png",
    "umbreon":   "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/197.png",
    "ralts":     "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/280.png",
    "gardevoir": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/282.png",
    "charmander":"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/4.png",
    "torchic":   "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/255.png",
}

headers = {"User-Agent": "Mozilla/5.0"}

for name, url in POKEMON_URLS.items():
    save_path = os.path.join(ASSET_DIR, f"{name}.png")
    if os.path.exists(save_path):
        print(f"[SKIP] {name}.png already exists")
        continue
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(save_path, "wb") as f:
            f.write(data)
        print(f"[OK]   {name}.png  ({len(data)//1024} KB)")
    except Exception as e:
        print(f"[ERR]  {name}: {e}")

print("\n완료! assets/ 폴더를 확인하세요.")
