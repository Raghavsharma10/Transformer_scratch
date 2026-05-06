def add_subparsers(self, *args, **kwargs):
        """
        Add subparsers to this parser

        Parameters
        ----------
        ``*args, **kwargs``
            As specified by the original
            :meth:`argparse.ArgumentParser.add_subparsers` method
        chain: bool
            Default: False. If True, It is enabled to chain subparsers"""
        chain = kwargs.pop('chain', None)
        ret = super(FuncArgParser, self).add_subparsers(*args, **kwargs)
        if chain:
            self._chain_subparsers = True
        self._subparsers_action = ret
        return ret