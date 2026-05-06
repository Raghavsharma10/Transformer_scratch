def make_parser(self, ctx):
        """Creates the underlying option parser for this command."""
        parser = OptionParser(ctx)
        parser.allow_interspersed_args = ctx.allow_interspersed_args
        parser.ignore_unknown_options = ctx.ignore_unknown_options
        for param in self.get_params(ctx):
            param.add_to_parser(parser, ctx)
        return parser