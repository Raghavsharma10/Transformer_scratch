def prepare_handler(cfg):
    """
    Load all files into single object.
    """
    positions, velocities, box = None, None, None
    _path = cfg.get('_path', './')
    forcefield = cfg.pop('forcefield', None)
    topology_args = sanitize_args_for_file(cfg.pop('topology'), _path)

    if 'checkpoint' in cfg:
        restart_args = sanitize_args_for_file(cfg.pop('checkpoint'), _path)
        restart = Restart.load(*restart_args)
        positions = restart.positions
        velocities = restart.velocities
        box = restart.box

    if 'positions' in cfg:
        positions_args = sanitize_args_for_file(cfg.pop('positions'), _path)
        positions = Positions.load(*positions_args)
        box = BoxVectors.load(*positions_args)

    if 'velocities' in cfg:
        velocities_args = sanitize_args_for_file(cfg.pop('velocities'), _path)
        velocities = Velocities.load(*velocities_args)

    if 'box' in cfg:
        box_args = sanitize_args_for_file(cfg.pop('box'), _path)
        box = BoxVectors.load(*box_args)

    options = {}
    for key in 'positions velocities box forcefield'.split():
        value = locals()[key]
        if value is not None:
            options[key] = value

    return SystemHandler.load(*topology_args, **options)