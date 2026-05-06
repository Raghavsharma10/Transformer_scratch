def archive_experiment(experiment):
    """Archives an experiment"""
    redis = oz.redis.create_connection()
    oz.bandit.Experiment(redis, experiment).archive()