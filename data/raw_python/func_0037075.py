def cli(
        ctx,
        config_file=None,
        requirements=None,
        profile=None):
    '''
    An abstraction layer for data storage systems

    DataFS is a package manager for data. It manages file versions,
    dependencies, and metadata for individual use or large organizations.

    For more information, see the docs at https://datafs.readthedocs.io
    '''

    ctx.obj = _DataFSInterface()

    ctx.obj.config_file = config_file
    ctx.obj.requirements = requirements
    ctx.obj.profile = profile

    def teardown():
        if hasattr(ctx.obj, 'api'):
            ctx.obj.api.close()

    ctx.call_on_close(teardown)