def cli(ctx, config, no_color):
    """ Command line client for lightflow. A lightweight, high performance pipeline
    system for synchrotrons.

    Lightflow is being developed at the Australian Synchrotron.
    """
    ctx.obj = {
        'show_color': not no_color if no_color is not None else True,
        'config_path': config
    }