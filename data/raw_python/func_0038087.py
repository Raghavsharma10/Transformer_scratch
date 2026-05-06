def serve(config):
    """
    Serves (runs) the vodka application
    """

    cfg = vodka.config.Config(read=config)
    vodka.run(cfg, cfg)