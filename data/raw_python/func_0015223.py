def add_argument_to(self, parser):
        """Used by cli to add this as an argument to argparse parser.

        Args:
            parser: parser to add this argument to
        """
        from devassistant.cli.devassistant_argparse import DefaultIffUsedActionFactory
        if isinstance(self.kwargs.get('action', ''), list):
            # see documentation of DefaultIffUsedActionFactory to see why this is necessary
            if self.kwargs['action'][0] == 'default_iff_used':
                self.kwargs['action'] = DefaultIffUsedActionFactory.generate_action(
                    self.kwargs['action'][1])
        # In cli 'preserved' is not supported.
        # It needs to be removed because it is unknown for argparse.
        self.kwargs.pop('preserved', None)
        try:
            parser.add_argument(*self.flags, **self.kwargs)
        except Exception as ex:
            problem = "Error while adding argument '{name}': {error}".\
                format(name=self.name, error=repr(ex))
            raise exceptions.ExecutionException(problem)