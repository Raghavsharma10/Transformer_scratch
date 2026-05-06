def setup_config(command, filename, section, vars):
    """Place any commands to setup chatapp here"""
    confuri = "config:" + filename
    if ":" in section:
        confuri += "#" + section.rsplit(":", 1)[-1]
    conf = appconfig(confuri)

    load_environment(conf.global_conf, conf.local_conf)