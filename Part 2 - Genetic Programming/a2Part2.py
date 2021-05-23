import operator
import random
import pandas as pd
import numpy as np
from deap import algorithms, gp, base, creator, tools
import pygraphviz as pgv
from deap.gp import graph


# This code is built upon the basic framework from https://github.com/DEAP/deap/blob/master/examples/gp/symbreg_numpy.py
# and the graphing functions I copied from https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
# With adjustments to various functions and methods

def protected_division(left, right):
    # Note that I have left this in here for posterity, having decided to exclude it from my final function set
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


# The following block of code adds all my functions and terminals to the primitive set
primitive_set = gp.PrimitiveSet("MAIN", 1)
primitive_set.addPrimitive(np.add, 2, name="add")
primitive_set.addPrimitive(np.subtract, 2, name="sub")
primitive_set.addPrimitive(np.multiply, 2, name="mul")
# primitive_set.addPrimitive(protected_division, 2)
primitive_set.addPrimitive(np.negative, 1, name="neg")
primitive_set.addEphemeralConstant("rand101", lambda: random.randint(0, 4))
primitive_set.renameArguments(ARG0='x')

# Create the individual and the fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Add all the relevant functions and definitions to the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=primitive_set)

# Load the necessary data in from the file regression.txt
training_data = pd.read_csv('Data/regression.txt', delim_whitespace=True)
training_data = training_data.iloc[1:]
X_data = (np.array(training_data.x.values.tolist(), dtype=float))
Y_data = np.array(training_data.y.values.tolist())


def rmse_evaluation(individual):
    # Calculate the RMSE for this individual
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the root mean squared error (RMSE) between the expression
    # and the real function values
    diff = np.sqrt((np.sum((func(X_data) - Y_data) ** 2)) / len(X_data))
    return diff,


# Set the relevant parameters for the evaluation function, the selection method and so on
toolbox.register("evaluate", rmse_evaluation)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=10))
toolbox.decorate("expr_mut", gp.staticLimit(key=operator.attrgetter('height'), max_value=10))


if __name__ == '__main__':
    pop = toolbox.population(n=1000)  # Initial population of 1000
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats, halloffame=hof)

    print("This is the best equation from this run: " + str(hof[0]))
    print("This is the RMSE of this equation: " + str(hof[0].fitness))

    # Stuff below here is my code to draw the graph for the individual expression
    nodes, edges, labels = graph(hof[0])
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    print("This equation is represented in 'tree.pdf'")
    g.draw("tree.pdf")
