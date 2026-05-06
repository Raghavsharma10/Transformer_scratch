def protocol(handler, cfg):
    """
    Run all the stages in protocol

    Parameters
    ----------
    handler : SystemHandler
        Container of initial conditions of simulation

    cfg : dict
        Imported YAML file.
    """
    # Stages
    if 'stages' not in cfg:
        raise ValueError('Protocol must include stages of simulation')

    pos, vel, box = handler.positions, handler.velocities, handler.box
    stages = cfg.pop('stages')
    for stage_options in stages:
        options = DEFAULT_OPTIONS.copy()
        options.update(cfg)
        stage_system_options = prepare_system_options(stage_options)
        options.update(stage_options)
        options['system_options'].update(stage_system_options)
        stage = Stage(handler, positions=pos, velocities=vel, box=box,
                      total_stages=len(stages), **options)
        pos, vel, box = stage.run()
        del stage