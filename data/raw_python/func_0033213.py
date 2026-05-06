def remove_experiment_choice(experiment, choice):
    """Removes an experiment choice"""
    redis = oz.redis.create_connection()
    oz.bandit.Experiment(redis, experiment).remove_choice(choice)