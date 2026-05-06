def setup_app(command, conf, vars):
    """Place any commands to setup tg2raptorized here"""
    load_environment(conf.global_conf, conf.local_conf)
    setup_schema(command, conf, vars)
    bootstrap.bootstrap(command, conf, vars)