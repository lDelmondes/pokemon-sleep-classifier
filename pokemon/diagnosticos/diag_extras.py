from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "raw" / "catalogo_mvp.csv", dtype={"numero": str})

faltam = ["45","85","114","124","150","190","191","192","213","214","215"]
svp = df[df["set_codigo"] == "SVP"]
extras = svp[svp["numero"].isin(faltam)].sort_values("numero")
print(extras[["numero", "nome", "image_url"]].to_string(index=False))