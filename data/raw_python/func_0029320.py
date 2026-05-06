def add_plugin_arguments(self, parser):
        """Add plugin arguments to argument parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The main haas ArgumentParser.

        """
        for manager in self.hook_managers.values():
            if len(list(manager)) == 0:
                continue
            manager.map(self._add_hook_extension_arguments, parser)
        for namespace, manager in self.driver_managers.items():
            choices = list(sorted(manager.names()))
            if len(choices) == 0:
                continue
            option, dest = self._namespace_to_option(namespace)
            parser.add_argument(
                option, help=self._help[namespace], dest=dest,
                choices=choices, default='default')
            option_prefix = '{0}-'.format(option)
            dest_prefix = '{0}_'.format(dest)
            manager.map(self._add_driver_extension_arguments,
                        parser, option_prefix, dest_prefix)