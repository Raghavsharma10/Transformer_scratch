def check_config(config):
    """
    Check and validate configuration attributes, to help administrators
    quickly spot missing required configurations and invalid configuration
    values in general
    """


    cfg = vodka.config.Config(read=config)

    vodka.log.set_loggers(cfg.get("logging"))
    vodka.app.load_all(cfg)

    click.echo("Checking config at %s for errors ..." % config)

    num_crit, num_warn = vodka.config.InstanceHandler.validate(cfg)

    click.echo("%d config ERRORS, %d config WARNINGS" % (num_crit, num_warn))