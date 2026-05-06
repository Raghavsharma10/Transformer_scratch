def app_cache_restorer():
    """
    A context manager that restore model cache state as it was before
    entering context.
    """
    state = _app_cache_deepcopy(apps.__dict__)
    try:
        yield state
    finally:
        with apps_lock():
            apps.__dict__ = state
            # Rebind the app registry models cache to
            # individual app config ones.
            for app_conf in apps.get_app_configs():
                app_conf.models = apps.all_models[app_conf.label]
            apps.clear_cache()