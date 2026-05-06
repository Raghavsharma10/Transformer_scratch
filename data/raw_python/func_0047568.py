def add_options(cls, manager):
        """Register plug-in specific options."""
        kw = {}
        if flake8.__version__ >= '3.0.0':
            kw['parse_from_config'] = True
        manager.add_option(
            "--known-modules",
            action='store',
            default="",
            help=(
                "User defined mapping between a project name and a list of"
                " provided modules. For example: ``--known-modules=project:"
                "[Project],extra-project:[extras,utilities]``."
            ),
            **kw
        )