import spotpy
import numpy as np


def run_spotpy(parameter_dict, runs):

    class spotpy_setup(object):

        # set parameter range
        par_dict = {}
        for k, v in parameter_dict.items():
            exec("%s_list = spotpy.parameter.Uniform(low = %s, high = %s)" % (k, v[0], v[1]))

        def __init__(self):
            pass

        def evaluation(self):
            pass

        def simulation(self, par):
            if par[-1] == 0:
                par[-1] = True
            else:
                par[-1] = False

            return par

        def objectivefunction(self, simulation, evaluation):
            return 1

    sampler = spotpy.algorithms.lhs(spotpy_setup(), dbname='parameter', dbformat='ram')

    # Run the sampler for a given number of repetitions
    sampler.sample(runs)
    parameter = sampler.getdata()

    return parameter

