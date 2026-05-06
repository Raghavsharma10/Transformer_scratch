def _dispatch_gen(self):
        """
        Process the generate subset of commands.
        """

        if not os.path.isdir(self._args.output):
            raise exception.Base("%s is not a writeable directory" % self._args.output)

        if not os.path.isfile(self._args.models_definition):
            if not self.check_package_exists(self._args.models_definition):
                raise exception.Base("failed to locate package or models definitions file at: %s" % self._args.models_definition)

        from prestans.devel.gen import Preplate
        preplate = Preplate(
            template_type=self._args.template,
            models_definition=self._args.models_definition,
            namespace=self._args.namespace,
            filter_namespace=self._args.filter_namespace,
            output_directory=self._args.output)

        preplate.run()