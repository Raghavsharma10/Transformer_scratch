def spawn(cls, options=None, dir_base=None):
        """Alternative constructor. Creates a mutator and returns section object.

        :param dict options:
        :param str|unicode dir_base:

        :rtype: SectionMutator

        """
        from uwsgiconf.utils import ConfModule

        options = options or {
            'compile': True,
        }

        dir_base = os.path.abspath(dir_base or find_project_dir())

        name_module = ConfModule.default_name
        name_project = get_project_name(dir_base)
        path_conf = os.path.join(dir_base, name_module)

        if os.path.exists(path_conf):
            # Read an existing config for further modification of first section.
            section = cls._get_section_existing(name_module, name_project)

        else:
            # Create section on-fly.
            section = cls._get_section_new(dir_base)

        mutator = cls(
            section=section,
            dir_base=dir_base,
            project_name=name_project,
            options=options)

        mutator.mutate()

        return mutator