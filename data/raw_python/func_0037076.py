def configure(ctx, helper, edit):
    '''
    Update configuration
    '''

    ctx.obj.config = ConfigFile(ctx.obj.config_file)

    if edit:
        ctx.obj.config.edit_config_file()
        return

    if os.path.isfile(ctx.obj.config.config_file):
        ctx.obj.config.read_config()

    if ctx.obj.profile is None:
        ctx.obj.profile = ctx.obj.config.default_profile

    args, kwargs = _parse_args_and_kwargs(ctx.args)
    assert len(args) == 0, 'Unrecognized arguments: "{}"'.format(args)

    if ctx.obj.profile not in ctx.obj.config.config['profiles']:
        ctx.obj.config.config['profiles'][ctx.obj.profile] = {
            'api': {'user_config': {}}, 'manager': {}, 'authorities': {}}

    profile_config = ctx.obj.config.config['profiles'][ctx.obj.profile]
    profile_config['api']['user_config'].update(kwargs)

    ctx.obj.config.write_config(ctx.obj.config_file)

    _generate_api(ctx)

    if ctx.obj.api.manager is not None:
        check_requirements(
            to_populate=profile_config['api']['user_config'],
            prompts=ctx.obj.api.manager.required_user_config,
            helper=helper)

    ctx.obj.config.write_config(ctx.obj.config_file)