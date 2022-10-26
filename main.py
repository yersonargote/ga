import random

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools

clausulas = []


def leer_archivo(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        vars, clause = tuple(map(int, lines[0].split(" ")))
        # print(vars, clause)
        lines = lines[1:]
        clausulas = []
        for line in lines:
            values = np.array(list(map(int, line.split(" ")[:-1])))
            # print(values, type(values))
            clausulas.append(values)
    return vars, clause, np.array(clausulas)


def plot_evolucion(log):
    """
    Representa la evolución del mejor individuo en cada generación
    """
    print(f"Max {log.select('max')} len {len(log.select('max'))}")
    print(f"Min {log.select('min')} len {len(log.select('min'))}")
    print(f"Gen {log.select('gen')} len {len(log.select('gen'))}")

    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    _, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    # ax1.fill_between(
    #     gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor="g", alpha=0.2
    # )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    # ax1.set_ylim([-10, 160])
    ax1.legend(["Min", "Max", "Avg"], loc="lower center")
    plt.grid(True)
    plt.savefig("Convergencia.eps", dpi=300)

    plt.show()


def funcion_objetivo(x):
    global clausulas
    sol = []
    lst = []
    for clausula in clausulas:
        for var in clausula:
            idx = abs(var) - 1
            value = x[idx]
            if var < 0:
                value = 1 - value
            sol.append(value)
        if 1 in sol:
            lst.append(1)
    sum = np.sum(np.array(lst))
    return (sum,)


def definicion_problema(size):
    # Creamos los objetos para definir el problema y el tipo de individuo
    creator.create(
        "FitnessMax", base.Fitness, weights=(1.0,)
    )  # el 1.0 se coloca para maximizar y para minimizar se coloca -1.0
    creator.create(
        "Individual", list, fitness=creator.FitnessMax
    )  # creator.FitnessMax viene de la linea anterior

    toolbox = base.Toolbox()

    # Generación de genes
    toolbox.register("attr_bin", random.randint, 0, 1)  # solo genera 1s y 0s
    # toolbox.register("attr_bin", random.uniform, 0, 1)

    # Generación de inviduos y población
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_bin, size
    )  # toolbox.attr_bin viene de la linea anterior # 16 viene de la cantidad inicial de individuos
    toolbox.register(
        "population", tools.initRepeat, list, toolbox.individual, 30
    )  # toolbox.individual viene de la linea anterior # 30 es el numero de repeticiones

    # Registro de operaciones genéticas
    toolbox.register("evaluate", funcion_objetivo)  # se coloca la funcion objetivo
    toolbox.register(
        "mate", tools.cxOnePoint
    )  # se utiliza un operador de cruze de un solo punto
    toolbox.register(
        "mutate", tools.mutFlipBit, indpb=0.05
    )  # se utiliza un operador de mutacion
    toolbox.register(
        "select", tools.selTournament, tournsize=3
    )  # se utiliza el metodo de seleccion de torneo, se seleccionan 3 individuos de forma aleatoria
    return toolbox


def main():
    global clausulas
    size, _, clausulas = leer_archivo("conf.txt")
    toolbox = definicion_problema(
        size,
    )
    random.seed(42)
    CXPB, MUTPB, NGEN = (
        0.5,
        0.2,
        20,
    )  # probabilidad de cruce, probabilidad de mutacion, numero de generaciones
    pop = toolbox.population()
    hof = tools.HallOfFame(1)  # se almacena el mejor
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )  # usa el algoritmo genetico simple
    return hof, logbook


if __name__ == "__main__":
    best, log = main()
    print("Mejor fitness: %f" % best[0].fitness.values)
    print("Mejor individuo %s" % best[0])
    plot_evolucion(log)
