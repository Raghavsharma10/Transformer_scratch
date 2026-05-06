def add_experiment(experiment):
    """Adds a new experiment"""
    redis = oz.redis.create_connection()
    oz.bandit.add_experiment(redis, experiment)