def var(ctx, clear_target, clear_all):
    """Install variable data to /var/[lib,cache]/hfos"""

    install_var(str(ctx.obj['instance']), clear_target, clear_all)