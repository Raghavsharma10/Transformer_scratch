def auto_sync(instance):
    """Returns bool if auto_sync is on for the model (instance)"""
    # this allows us to turn off sync temporarily - e.g. when doing bulk updates
    if not get_setting("auto_sync"):
        return False
    model_name = "{}.{}".format(instance._meta.app_label, instance._meta.model_name)
    if model_name in get_setting("never_auto_sync", []):
        return False
    return True