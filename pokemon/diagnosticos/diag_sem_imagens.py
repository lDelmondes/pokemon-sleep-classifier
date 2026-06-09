from pokemon.caminhos import RAW
import pandas as pd

def main():
    df = pd.read_csv(RAW / "catalogo_completo.csv", dtype={"numero": str})
    sem_img = df[df["image_url"].isna()]
    com_img = df[df["image_url"].notna()]

    print(f"Total no catalogo:  {len(df)}")
    print(f"  com imagem:       {len(com_img)}  -> pipeline automatico")
    print(f"  sem imagem:       {len(sem_img)}  -> ponto cego do modelo")

    print(f"\nSem imagem por set:")
    print(sem_img["set_codigo"].value_counts().to_string())

    print(f"\nAs {len(sem_img)} cartas sem imagem:")
    print(sem_img[["card_id", "set_codigo", "numero", "nome"]]
          .sort_values(["set_codigo", "numero"]).to_string(index=False))

if __name__ == "__main__":
    main()