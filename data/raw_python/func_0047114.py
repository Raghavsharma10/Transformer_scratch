def _set_jinja2_enviroment(self):
        """
        Set up the jinja2 environment.
        """

        template_loader = FileSystemLoader(searchpath=self.TEMPLATE_DIR)

        env = Environment(loader=template_loader, trim_blocks=True,
                          lstrip_blocks=True)
        env.globals.update(chunker=chunker, enumerate=enumerate, str = str)

        # Add filters to the environment
        round2digits = functools.partial(round_to_nearest, nearest=0.1)
        env.filters['round2digits'] = round2digits
        env.filters['mean'] = statistics.mean

        self.jinja2_environment = env