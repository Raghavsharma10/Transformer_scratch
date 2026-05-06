def add_experiment_choice(experiment, choice):
    """Adds an experiment choice"""
    redis = oz.redis.create_connection()
    oz.bandit.Experiment(redis, experiment).add_choice(choice)