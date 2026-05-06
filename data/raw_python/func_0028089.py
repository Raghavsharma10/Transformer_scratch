def pref_init():
    """Can be called without penalty. Create ~/.cdk dir if it doesn't
    exist. Copy the default pref file if it doesn't exist."""

    # make sure we have a ~/.cdk dir
    if not isdir(PREFS_DIR):
        mkdir(PREFS_DIR)
    # make sure we have a default prefs file
    if not isfile(PREFS_FILE):
        copy(join(LOCATION, "custom", "prefs"), PREFS_DIR)