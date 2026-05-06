def sync_experiments_from_spec(filename):
    """
    Takes the path to a JSON file declaring experiment specifications, and
    modifies the experiments stored in redis to match the spec.

    A spec looks like this:
    {
       "experiment 1": ["choice 1", "choice 2", "choice 3"],
       "experiment 2": ["choice 1", "choice 2"]
    }
    """

    redis = oz.redis.create_connection()

    with open(filename, "r") as f:
        schema = escape.json_decode(f.read())

    oz.bandit.sync_from_spec(redis, schema)