import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------
# CARGA DE PARÁMETROS (Punto 6)
# ---------------------------------------------------------
with open('config.json', 'r') as f:
    config = json.load(f)

SEED = config["semilla_aleatoria"]
POP_SIZE = config["tamano_poblacion"]
GENERATIONS = config["numero_generaciones"]
P_CROSSOVER = config["probabilidad_cruce"]
P_MUTATION = config["probabilidad_mutacion"]
CHROMOSOME_LENGTH = config["longitud_cromosoma"]

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------
# FUNCIONES DEL ALGORITMO GENÉTICO (Puntos 1 al 5)
# ---------------------------------------------------------
def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 1

def decode_chromosome(chromosome, bounds=(-1, 2)):
    min_val, max_val = bounds
    integer_val = int("".join(str(bit) for bit in chromosome), 2)
    max_integer = (2 ** CHROMOSOME_LENGTH) - 1
    return min_val + (integer_val / max_integer) * (max_val - min_val)

def create_individual():
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]

def tournament_selection(population, fitnesses, k=3):
    selected_indices = random.sample(range(len(population)), k)
    best_index = max(selected_indices, key=lambda i: fitnesses[i])
    return population[best_index]

def one_point_crossover(parent1, parent2):
    if random.random() < P_CROSSOVER:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1[:], parent2[:]

def mutate(chromosome):
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < P_MUTATION:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def brute_force_validation():
    x_vals = np.linspace(-1, 2, 10000)
    y_vals = fitness_function(x_vals)
    best_idx = np.argmax(y_vals)
    return x_vals[best_idx], y_vals[best_idx]

# ---------------------------------------------------------
# INTERFAZ GRÁFICA (DASHBOARD ALINEADO A LA RÚBRICA)
# ---------------------------------------------------------
def mostrar_interfaz(best_ind, best_x, best_fit, bf_x, bf_y, error, history):
    root = tk.Tk()
    root.title("Entregable - Algoritmo Genético")
    root.geometry("1000x550")
    root.configure(bg="#F8F9FA") 
    
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background="#F8F9FA")
    style.configure("TLabel", background="#F8F9FA", font=("Segoe UI", 10), foreground="#333333")
    style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#2C3E50")
    
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # --- PANEL IZQUIERDO: RÚBRICA Y DATOS ---
    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
    
    # Sección 1: Evidencia de los Puntos 1 al 5
    ttk.Label(left_frame, text="⚙️ Configuración del Modelo", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
    config_text = (
        "• Representación: Binaria (22 bits)\n"
        "• Selección: Torneo\n"
        "• Cruce: Un punto\n"
        "• Mutación: Bit-flip\n"
        "• Función: f(x) = x·sin(10πx) + 1"
    )
    ttk.Label(left_frame, text=config_text, foreground="#555555").pack(anchor=tk.W, pady=(0, 15))

    # Sección 2: Resultados del AG (Punto 7)
    ttk.Label(left_frame, text="🏆 Mejor Individuo (AG)", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
    ttk.Label(left_frame, text=f"Cromosoma:\n{best_ind}").pack(anchor=tk.W, pady=2)
    ttk.Label(left_frame, text=f"Fenotipo (x): {best_x:.6f}").pack(anchor=tk.W, pady=2)
    ttk.Label(left_frame, text=f"Aptitud máxima: {best_fit:.6f}").pack(anchor=tk.W, pady=(2, 15))
    
    # Sección 3: Validación (Punto 8)
    ttk.Label(left_frame, text="🔍 Validación Analítica", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
    ttk.Label(left_frame, text=f"Óptimo real (x): {bf_x:.6f}").pack(anchor=tk.W, pady=2)
    ttk.Label(left_frame, text=f"Óptimo real f(x): {bf_y:.6f}").pack(anchor=tk.W, pady=2)
    
    # Recuadro de Error
    error_frame = tk.Frame(left_frame, bg="#FFFFFF", highlightbackground="#D3D3D3", highlightthickness=1, padx=10, pady=8)
    error_frame.pack(anchor=tk.W, pady=15, fill=tk.X)
    tk.Label(error_frame, text="Error vs Óptimo:", bg="#FFFFFF", fg="#666666", font=("Segoe UI", 9)).pack(anchor=tk.W)
    tk.Label(error_frame, text=f"{error:.6e}", bg="#FFFFFF", fg="#E74C3C", font=("Consolas", 11, "bold")).pack(anchor=tk.W)
    
    ttk.Button(left_frame, text="Cerrar Panel", command=root.destroy).pack(side=tk.BOTTOM, anchor=tk.W, pady=10)
    
    # --- PANEL DERECHO: GRÁFICA (Punto 7) ---
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    ax.plot(range(len(history)), history, marker='o', color='#2980B9', linestyle='-', markersize=4, linewidth=1.5)
    ax.set_title('Convergencia del Algoritmo Genético', fontsize=12, color='#2C3E50', weight='bold')
    ax.set_xlabel('Generación', color='#555555')
    ax.set_ylabel('Mejor Aptitud f(x)', color='#555555')
    ax.grid(True, linestyle=':', alpha=0.7, color='#A6ACAF')
    
    fig.tight_layout()
    fig.savefig('convergencia.png') # Guarda el entregable
    
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    root.mainloop()

# ---------------------------------------------------------
# CICLO PRINCIPAL
# ---------------------------------------------------------
def main():
    population = [create_individual() for _ in range(POP_SIZE)]
    history_best_fitness = []
    
    best_overall_individual = None
    best_overall_fitness = -float('inf')
    best_overall_x = None

    for _ in range(GENERATIONS):
        real_values = [decode_chromosome(ind) for ind in population]
        fitnesses = [fitness_function(x) for x in real_values]
        
        gen_best_fitness = max(fitnesses)
        gen_best_idx = fitnesses.index(gen_best_fitness)
        history_best_fitness.append(gen_best_fitness)
        
        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_individual = population[gen_best_idx]
            best_overall_x = real_values[gen_best_idx]

        new_population = []
        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = one_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.extend([c1, c2])
            
        population = new_population[:POP_SIZE]

    bf_x, bf_y = brute_force_validation()
    error = abs(bf_y - best_overall_fitness)
    
    mostrar_interfaz(best_overall_individual, best_overall_x, best_overall_fitness, bf_x, bf_y, error, history_best_fitness)

if __name__ == "__main__":
    main()