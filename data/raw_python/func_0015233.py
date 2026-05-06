def _run_path_dependencies(self, parsed_args):
        """Installs dependencies from the leaf assistant.
        Raises:
            devassistant.exceptions.DependencyException with a cause if something goes wrong
        """
        deps = self.path[-1].dependencies(parsed_args)

        lang.Command('dependencies', deps, parsed_args).run()