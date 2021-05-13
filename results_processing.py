import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_path = 'results/results.csv'
    df = pd.read_csv(file_path)

    charts_dir = 'data/charts/'

    problem_size = df['size']
    optimal_solution = df['optimal_solution']
    bees_result = df['bees_result']
    genetic_result = df['genetic_result']

    # solutions comparison #  w sumie to chyba nie potrzebny ten wykres, bo i tak nic nie mówi
    plt.title("Solutions")
    plt.plot(problem_size, optimal_solution)
    plt.plot(problem_size, bees_result)
    plt.plot(problem_size, genetic_result)
    plt.xlabel("Problem size [number of variables]")
    plt.ylabel("Distance")
    plt.legend(['Optimal', 'Bees', 'Genetic'])

    plt.savefig(charts_dir + 'solutions.svg', format='svg')
    plt.show()

    # error of solution # ten tak samo
    plt.clf()

    bees_error = [b - o for b, o in zip(bees_result, optimal_solution)]
    genetic_error = [g - o for g, o in zip(genetic_result, optimal_solution)]

    plt.title("Error")
    plt.plot(problem_size, bees_error)
    plt.plot(problem_size, genetic_error)
    plt.xlabel("Problem size [number of variables]")
    plt.ylabel("Error of solution")
    plt.legend(['Bees', 'Genetic'])

    plt.savefig(charts_dir + 'error.svg', format='svg')

    plt.show()

    # % error
    plt.clf()

    bees_error_percentage = [round(((b / o) - 1) * 100, 2) for b, o in zip(bees_result, optimal_solution)]
    genetic_error_percentage = [round(((g / o) - 1) * 100, 2) for g, o in zip(genetic_result, optimal_solution)]

    plt.title("% of error")
    plt.plot(problem_size, bees_error_percentage)
    plt.plot(problem_size, genetic_error_percentage)
    plt.xlabel("Problem size [number of variables]")
    plt.ylabel("% of error")
    plt.legend(['Bees', 'Genetic'])

    plt.savefig(charts_dir + 'error_percentage.svg', format='svg')

    plt.show()

    # processing time
    plt.clf()

    bees_processing_time = df['bees_time']
    genetic_processing_time = df['genetic_time']

    plt.title("Processing time")
    plt.plot(problem_size, bees_processing_time)
    plt.plot(problem_size, genetic_processing_time)
    plt.xlabel("Problem size [number of variables]")
    plt.ylabel("Time [s]")
    plt.legend(['Bees', 'Genetic'])

    plt.savefig(charts_dir + 'processing_time.svg', format='svg')

    plt.show()



