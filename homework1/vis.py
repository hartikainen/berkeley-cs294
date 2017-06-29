from itertools import groupby
import json
import argparse

import matplotlib
# matplotlib.use(args.mpl_backend)
import matplotlib.pyplot as plt

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

    parser.add_argument("--num_rollouts",
                        type=int,
                        default=20,
                        help="Number of expert roll outs")

    parser.add_argument("--dagger_N",
                        type=int,
                        default=10,
                        help=("Number of dagger iterations."))

    args = vars(parser.parse_args())

    return args


def plot_behavioral_cloning(bc_results):
    bc_results = sorted(
        sorted(bc_results, key=lambda x: x['num_rollouts']),
        key=lambda x: x['env']
    )


    for env, results in groupby(bc_results, lambda r: r['env']):
        rollouts, mean_returns, std_returns = zip(*[
            (r['num_rollouts'], np.mean(r['returns']), np.std(r['returns']))
            for r in results
        ])

        plt.errorbar(rollouts, mean_returns, yerr=std_returns,
                     fmt='-o', capsize=4, elinewidth=2, label=env)

    plt.legend(loc="upper right")
    plt.xlabel("mean returns", fontsize=16)
    plt.ylabel("number of rollouts", fontsize=16)
    plt.suptitle("Behavioral cloning mean rewards against expert rollouts",
                 fontsize=20)

    plt.show()


def filter_data(dagger_results, bc_results, expert_results):
    """ Filters and sorts the data to be used for visualization

    Input:
    - dagger_results, bc_results, expert_results: Arrays of containing results
      for the runs.
    - num_rollouts: number of rollouts to filter the data based on
    - dagger_N: number of dagger iterations to filter the data based on

    Returns:
    - dagger_results, expert_results, bc_results: filtered lists of same
      length, and each sorted by environment name
    """
    dagger_results = filter(lambda res: (res['num_rollouts'] == num_rollouts
                                         and res['dagger_N'] == dagger_N),
                            dagger_results)
    dagger_results = sorted(dagger_results, key=lambda res: res['env'])

    expert_results = filter(lambda res: res['num_rollouts'] == num_rollouts,
                            expert_results)
    expert_results = sorted(expert_results, key=lambda res: res['envname'])
    bc_results = filter(lambda res: res['num_rollouts'] == num_rollouts,

                        bc_results)
    bc_results = sorted(bc_results, key=lambda res: res['env'])

    return dagger_results, expert_results, bc_results


def plot_dagger(dagger_results, bc_results, expert_results,
                num_rollouts, dagger_N):
    """ DAgger results plot with comparison against BC and expert results

    Input:
    - dagger_results: Array of length N, containing results for different dagger
      runs.
    - bc_results: Array of length N, containing results for behavioral cloning,
      correspoding to the dagger_results
    - expert_results: Array of length N, containing results for expert,
      correspoding to the dagger_results
    """
    dagger_results, bc_results, expert_results = filter_data(dagger_results,
                                                             bc_results,
                                                             expert_results,
                                                             num_rollouts,
                                                             dagger_N)

    for dagger_result, bc_result, expert_result in zip(dagger_results,
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
    num_rollouts = args['num_rollouts']
    dagger_N = args['dagger_N']

    with open(args['dagger_results'], 'r') as f:
        dagger_results = json.load(f)

    with open(args['expert_results'], 'r') as f:
        expert_results = json.load(f)

    with open(args['bc_results'], 'r') as f:
        bc_results = json.load(f)

    plot_behavioral_cloning(bc_results)

    assert(all((dr['env'] == bcr['env'] == er['envname']
                and (dr['num_rollouts']
                     == bcr['num_rollouts']
                     == er['num_rollouts']))
               for dr, bcr, er in zip(dagger_results,
                                      bc_results,
                                      expert_results)))

    plot_dagger(dagger_results, bc_results, expert_results,
                num_rollouts, dagger_N)
