def parse_denovo_params(user_params=None):
    """Return default GimmeMotifs parameters. 

    Defaults will be replaced with parameters defined in user_params.

    Parameters
    ----------
    user_params : dict, optional
        User-defined parameters.

    Returns
    -------
    params : dict
    """
    config = MotifConfig()

    if user_params is None:
        user_params = {}
    params = config.get_default_params()
    params.update(user_params)

    if params.get("torque"):
        logger.debug("Using torque")
    else:
        logger.debug("Using multiprocessing")

    params["background"] = [x.strip() for x in params["background"].split(",")]

    logger.debug("Parameters:")
    for param, value in params.items():
        logger.debug("  %s: %s", param, value)

    # Maximum time?
    
    if params["max_time"]:
        try:
            max_time = params["max_time"] = float(params["max_time"])
        except Exception:
            logger.debug("Could not parse max_time value, setting to no limit")
            params["max_time"] = -1

    if params["max_time"] > 0:
        logger.debug("Time limit for motif prediction: %0.2f hours", max_time)
        params["max_time"] = 3600 * params["max_time"]
        logger.debug("Max_time in seconds %0.0f", max_time)
    else:
        logger.debug("No time limit for motif prediction")

    return params