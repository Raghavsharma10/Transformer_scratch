def set_settings(key, value):
    """Set Hitman internal settings."""
    with Database("settings") as settings:
        if value in ['0', 'false', 'no', 'off', 'False']:
            del settings[key]
            print("Disabled setting")
        else:
            print(value)
            settings[key] = value
            print("Setting saved")