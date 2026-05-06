def get_experiments(redis, active=True):
    """Gets the full list of experiments"""

    key = ACTIVE_EXPERIMENTS_REDIS_KEY if active else ARCHIVED_EXPERIMENTS_REDIS_KEY
    return [Experiment(redis, escape.to_unicode(name)) for name in redis.smembers(key)]