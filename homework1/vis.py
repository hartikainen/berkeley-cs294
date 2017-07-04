from itertools import groupby
import json
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

def get_dagger_file(vis_dir, num_rollouts, env_name):
    vis_path = "{}/dagger-{}-{}.pdf".format(vis_dir, env_name, num_rollouts)
    return vis_path

def get_bc_file(vis_dir):
    vis_path = "{}/behavioral_cloning.pdf".format(vis_dir)
    return vis_path


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

    parser.add_argument("--bc_dir",
                        type=str,
                        help=("Directory to save the behavioral cloning "
                              "visualizations to"))

    parser.add_argument("--dagger_dir",
                        type=str,
                        help=("Directory to save the DAgger "
                              "visualizations to"))

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


def plot_behavioral_cloning(bc_results, vis_dir):
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
                     fmt='-o', capsize=4, elinewidth=1, label=env)

    plt.legend(loc="lower right", fontsize=6)
    plt.xlabel("mean returns", fontsize=16)
    plt.ylabel("number of rollouts", fontsize=16)

    if vis_dir is not None:
        vis_file = get_bc_file(vis_dir)
        plt.savefig(vis_file,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.show()


def filter_data(dagger_results, bc_results, expert_results,
                num_rollouts, dagger_N):
    """ Filters and sorts the data to be used for visualization

    Input:
    - dagger_results, bc_results, expert_results: Arrays of containing results
      for the runs.
    - num_rollouts: number of rollouts to filter the data based on
    - dagger_N: number of dagger iterations to filter the data based on

    Returns:
    - dagger_results, expert_results, bc_results: filtered lists, each of same
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

    return dagger_results, bc_results, expert_results


def plot_dagger(dagger_results, bc_results, expert_results,
                num_rollouts, dagger_N, vis_dir=None):
    """ DAgger results plot with comparison against BC and expert results
    """
    dagger_results, bc_results, expert_results = filter_data(dagger_results,
                                                             bc_results,
                                                             expert_results,
                                                             num_rollouts,
                                                             dagger_N)

    assert(all((dr['env'] == bcr['env'] == er['envname']
                and (dr['num_rollouts']
                     == bcr['num_rollouts']
                     == er['num_rollouts']
                     == num_rollouts))
               for dr, bcr, er in zip(dagger_results,
                                      bc_results,
                                      expert_results)))

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
                     fmt='-o', capsize=4, elinewidth=1)

        plt.axhline(y=np.mean(expert_result['returns']),
                    color='k',
                    label="Expert Policy")
        plt.axhline(y=np.mean(bc_result['returns']),
                    color='r',
                    label="Behaviorial Cloning")

        plt.xlabel("iteration", fontsize=16)
        plt.ylabel("mean reward", fontsize=16)
        plt.legend()

        if vis_dir is not None:
            vis_file = get_dagger_file(vis_dir, num_rollouts, env_name)
            plt.savefig(vis_file,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
        else:
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

    dagger_dir, bc_dir = args['dagger_dir'], args['bc_dir']

    plot_behavioral_cloning(bc_results, bc_dir)

    plot_dagger(dagger_results, bc_results, expert_results,
                num_rollouts, dagger_N, dagger_dir)
