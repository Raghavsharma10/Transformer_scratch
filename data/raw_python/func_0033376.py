def add_experiment(redis, name):
    """Adds a new experiment"""

    if not ALLOWED_NAMES.match(name):
        raise ExperimentException(name, "Illegal name")
    if redis.exists(EXPERIMENT_REDIS_KEY_TEMPLATE % name):
        raise ExperimentException(name, "Already exists")

    json = dict(creation_date=util.unicode_type(datetime.datetime.now()))
    pipe = redis.pipeline(transaction=True)
    pipe.sadd(ACTIVE_EXPERIMENTS_REDIS_KEY, name)
    pipe.hset(EXPERIMENT_REDIS_KEY_TEMPLATE % name, "metadata", escape.json_encode(json))
    pipe.execute()
    return Experiment(redis, name)