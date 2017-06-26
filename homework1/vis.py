import matplotlib
# matplotlib.use(args.mpl_backend)
import matplotlib.pyplot as plt
import json
import argparse

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plotting for Behavioral Cloning")

    parser.add_argument("--dagger_results",
                        type=str,
                        help="Path to the DAgger results file")

    parser.add_argument("--expert_results",
                        type=str,
                        help="Path to the expert results file")

    parser.add_argument("--bc_results",
                        type=str,
                        help="Path to the behavioral cloning results file")

    args = vars(parser.parse_args())

    return args


def plot_dagger(dagger_results, bc_results, expert_results):
    """ DAgger results plot with comparison against BC and expert results

    Input:
    - dagger_results: Array of length N, containing results for different dagger
      runs.
    - bc_results: Array of length N, containing results for behavioral cloning,
      correspoding to the dagger_results
    - expert_results: Array of length N, containing results for expert,
      correspoding to the dagger_results
    """
    for dagger_result, bc_result, expert_results in zip(dagger_results,
                                                        bc_results,
                                                        expert_results):
        env_name = dagger_result['env']
        expert_data_file = dagger_result['expert_data_file']

        fig = plt.figure()
        dagger_returns = dagger_result['returns']
        dagger_N = dagger_result['dagger_N']
        mean_returns = list(map(np.mean, dagger_returns))
        std_returns = list(map(np.std, dagger_returns))
        Ns = range(dagger_N)

        plt.errorbar(Ns, mean_returns, yerr=std_returns,
                     fmt='-o', capsize=4, elinewidth=2)
        plt.suptitle("DAgger rewards per iteration for {}"
                     "".format(env_name), fontsize=20)


        plt.axhline(y=np.mean(expert_result['returns']), color='k', label="Expert Policy")
        plt.axhline(y=np.mean(bc_result['returns']), color='r', label="Behaviorial Cloning")

        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward")
        plt.legend()

        plt.show()

    return

if __name__ == "__main__":

    args = parse_args()
    expert_result = [
        x for x in filter(lambda x: x['envname'] == env_name, expert_results)
    ][0]
    bc_result = [
        x for x in filter(lambda x: x['env'] == env_name, bc_results)
    ][0]
    with open(args['dagger_results'], 'r') as f:
        dagger_results = json.load(f)

    with open(args['expert_results'], 'r') as f:
        expert_results = json.load(f)

    with open(args['bc_results'], 'r') as f:
        bc_results = json.load(f)

    plot_dagger(dagger_results, bc_results, expert_results)
