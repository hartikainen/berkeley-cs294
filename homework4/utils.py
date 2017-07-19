import argparse

AVAILABLE_ENVS = (
    'Ant-v1',
    'HalfCheetah-v1',
    'Hopper-v1',
    'Humanoid-v1',
    'Reacher-v1',
    'Walker2d-v1'
)

# DEFAULT_ENV = 'Pendulum-v0'
DEFAULT_ENV = 'CartPole-v0'

def parse_args():
    parser = argparse.ArgumentParser(
        description="DAgger - Dataset Aggregation")

    parser.add_argument("--env",
                        type=str,
                        default=DEFAULT_ENV,
                        choices=AVAILABLE_ENVS,
                        help="The name of the environment")
    parser.add_argument("--max_timesteps",
                        type=int,
                        default=5e5,
                        help="Number timesteps to run")
    parser.add_argument("--ac_model",
                        type=str,
                        default="ActorCriticValueFullyConnected",
                        help=("Name of the class in models.py that will be "
                              "used as the actor critic model"))
    parser.add_argument("--agent",
                        type=str,
                        default="A3CAgent",
                        help=("Name of a function in agents.py that will be "
                              "used as the agent"))
    parser.add_argument("--N_parallel_learners",
                        type=int,
                        default=2,
                        help="Number of parallel learners")

    args = vars(parser.parse_args())

    return args
