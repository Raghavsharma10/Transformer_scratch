def action(self):
        """
        Invoke functions according to the supplied flags
        """

        user = self.args['--user'] if self.args['--user'] else None
        reset = True if self.args['--reset'] else False

        if self.args['generate']:
            generate_network(user, reset)
        elif self.args['publish']:
            publish_network(user, reset)