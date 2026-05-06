def trial(log_dir=None,
          upload_dir=None,
          sync_period=None,
          trial_prefix="",
          param_map=None,
          init_logging=True):
    """
    Generates a trial within a with context.
    """
    global _trial  # pylint: disable=global-statement
    if _trial:
        # TODO: would be nice to stack crawl at creation time to report
        # where that initial trial was created, and that creation line
        # info is helpful to keep around anyway.
        raise ValueError("A trial already exists in the current context")
    local_trial = Trial(
        log_dir=log_dir,
        upload_dir=upload_dir,
        sync_period=sync_period,
        trial_prefix=trial_prefix,
        param_map=param_map,
        init_logging=True)
    try:
        _trial = local_trial
        _trial.start()
        yield local_trial
    finally:
        _trial = None
        local_trial.close()