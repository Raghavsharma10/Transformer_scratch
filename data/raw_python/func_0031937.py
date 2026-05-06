def checkConfig():
    """If the config.py file exists, back it up"""
    config_file_dir = os.path.join(cwd, "config.py")
    if os.path.exists(config_file_dir):
        print("Making a backup of your config file!")
        config_file_dir2 = os.path.join(cwd, "config.py.oldbak")
        copyfile(config_file_dir, config_file_dir2)