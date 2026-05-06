def input(self, _in, out, **kw):
        """Input filtering."""
        args = [self.binary or 'cleancss'] + self.rebase_opt
        if self.extra_args:
            args.extend(self.extra_args)
        self.subprocess(args, out, _in)