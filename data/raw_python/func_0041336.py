def print_settings_example():
    """
    You can use settings to get additional information from the user via their
    dependencies.io configuration file. Settings will be automatically injected as
    env variables with the "SETTING_" prefix.

    All settings will be passed as strings. More complex types will be json
    encoded. You should always provide defaults, if possible.
    """
    SETTING_EXAMPLE_LIST = json.loads(os.getenv('SETTING_EXAMPLE_LIST', '[]'))
    SETTING_EXAMPLE_STRING = os.getenv('SETTING_EXAMPLE_STRING', 'default')

    print('List setting values: {}'.format(SETTING_EXAMPLE_LIST))
    print('String setting value: {}'.format(SETTING_EXAMPLE_STRING))