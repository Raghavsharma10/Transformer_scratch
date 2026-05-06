def parse_env(config_schema, env):
    """Parse the values from a given environment against a given config schema

    Args:
        config_schema: A dict which maps the variable name to a Schema object
            that describes the requested value.
        env: A dict which represents the value of each variable in the
            environment.
    """
    try:
        return {
            key: item_schema.parse(key, env.get(key))
            for key, item_schema in config_schema.items()
        }
    except KeyError as error:
        raise MissingConfigError(
            "Required config not set: {}".format(error.args[0])
        )