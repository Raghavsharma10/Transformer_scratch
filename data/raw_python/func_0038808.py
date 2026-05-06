def set(self, handler, attr, name, path, cfg):

        """
        Obtain value for config variable, by prompting the user
        for input and substituting a default value if needed.

        Also does validation on user input
        """

        full_name = ("%s.%s" % (path, name)).strip(".")

        # obtain default value
        if attr.default is None:
            default = None
        else:
            try:
                comp = vodka.component.Component(cfg)
                default = handler.default(name, inst=comp)
                if self.skip_defaults:
                    self.echo("%s: %s [default]" % (full_name, default))
                    return default
            except Exception:
                raise

        # render explanation
        self.echo("")
        self.echo(attr.help_text)
        if attr.choices:
            self.echo("choices: %s" % ", ".join([str(c) for c in attr.choices]))


        # obtain user input and validate until input is valid
        b = False
        while not b:
            try:
                if type(attr.expected_type) == type:
                    r = self.prompt(full_name, default=default, type=attr.expected_type)
                    r = attr.expected_type(r)
                else:
                    r = self.prompt(full_name, default=default, type=str)
            except ValueError:
                self.echo("Value expected to be of type %s"% attr.expected_type)
            try:
                b = handler.check({name:r}, name, path)
            except Exception as inst:
                if hasattr(inst, "explanation"):
                    self.echo(inst.explanation)
                else:
                    raise
        return r