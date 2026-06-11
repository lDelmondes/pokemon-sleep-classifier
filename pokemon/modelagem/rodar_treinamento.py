"""
Script Orquestrador para rodar múltiplos treinamentos em sequência.
Executa: python -m pokemon.modelagem.treinar_recorte_full <rodada>
"""
import subprocess
import time
import sys  # << Importação necessária para descobrir a .venv

RODADAS = 3

def main():
    print(f"Iniciando bateria automática de {RODADAS} treinamentos...")
    t_inicio_geral = time.perf_counter()

    # O "pulo do gato": Pega o caminho absoluto do Python da sua (.venv)
    # Ex: C:\Users\lucas\Documents\pokemon-sleep-classifier\.venv\Scripts\python.exe
    python_exe = sys.executable 

    for i in range(1, RODADAS + 1):
        print(f"\n{'='*40}")
        print(f"🚀 INICIANDO RODADA {i}/{RODADAS}")
        print(f"{'='*40}")
        
        t_inicio_rodada = time.perf_counter()

        comando = [
            python_exe,  # << Aqui paramos de usar a string genérica "python"
            "-m", 
            "pokemon.modelagem.treinar_recorte_full", 
            str(i)
        ]

        # Executa a rodada e espera terminar
        subprocess.run(comando)

        t_fim_rodada = time.perf_counter()
        duracao_minutos = (t_fim_rodada - t_inicio_rodada) / 60
        print(f"\n✅ Rodada {i} concluída em {duracao_minutos:.1f} minutos.")

    t_fim_geral = time.perf_counter()
    duracao_total = (t_fim_geral - t_inicio_geral) / 60
    print(f"\n🎉 Bateria finalizada! Tempo total: {duracao_total:.1f} minutos.")

if __name__ == "__main__":
    main()