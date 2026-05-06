def _get_section_existing(self, name_module, name_project):
        """Loads config section from existing configuration file (aka uwsgicfg.py)

        :param str|unicode name_module:
        :param str|unicode name_project:
        :rtype: Section

        """
        from importlib import import_module

        from uwsgiconf.settings import CONFIGS_MODULE_ATTR

        config = getattr(
            import_module(os.path.splitext(name_module)[0], package=name_project),
            CONFIGS_MODULE_ATTR)[0]

        return config.sections[0]