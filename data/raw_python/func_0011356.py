def setup(self):
        ''' obtain database and filesystem preferences from defaults,
            and compare with selection in container.
        '''

        self.selection = EXPFACTORY_EXPERIMENTS
        self.ordered = len(EXPFACTORY_EXPERIMENTS) > 0
        self.data_base = EXPFACTORY_DATA
        self.study_id = EXPFACTORY_SUBID
        self.base = EXPFACTORY_BASE
        self.randomize = EXPFACTORY_RANDOMIZE
        self.headless = EXPFACTORY_HEADLESS

        # Generate variables, if they exist
        self.vars = generate_runtime_vars() or None

        available = get_experiments("%s" % self.base)
        self.experiments = get_selection(available, self.selection)
        self.logger.debug(self.experiments)
        self.lookup = make_lookup(self.experiments)
        final = "\n".join(list(self.lookup.keys()))       

        bot.log("Headless mode: %s" % self.headless)
        bot.log("User has selected: %s" % self.selection)
        bot.log("Experiments Available: %s" %"\n".join(available))
        bot.log("Randomize: %s" % self.randomize)
        bot.log("Final Set \n%s" % final)