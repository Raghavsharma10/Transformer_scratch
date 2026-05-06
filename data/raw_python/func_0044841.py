def setup_config(command, filename, section, vars):
    """Place any commands to setup cogenircapp here"""
    conf = appconfig('config:' + filename)
    load_environment(conf.global_conf, conf.local_conf)