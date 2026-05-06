def analyze(self, args):
        """Run the analysis routines determined from the given `args`.
        """
        self.log.info("Running catalog analysis")

        if args.count:
            self.count()

        return