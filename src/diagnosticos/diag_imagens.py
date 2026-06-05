from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES = ROOT / "data" / "images"

arquivos = list(IMAGES.glob("*.png"))
print(f"{len(arquivos)} arquivos .png em disco.\n")

ok, quebradas = 0, 0
for f in arquivos:
    try:
        with Image.open(f) as img:
            img.verify()   # checa integridade sem carregar tudo na memoria
        ok += 1
    except Exception as e:
        quebradas += 1
        print(f"  QUEBRADA: {f.name} ({e})")

print(f"\nValidas: {ok} | Quebradas: {quebradas}")