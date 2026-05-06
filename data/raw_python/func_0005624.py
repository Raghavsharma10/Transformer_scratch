def run(self):
        """
        Run the application
        """
        self.call_plugins("on_run")
        if vars(self.arguments).get("version", None):
            self.logger.info("{app_name}: {version}".format(app_name=self.app_name, version=self.version))
        else:
            if self.arguments.command == "main":
                self.main()
            else:
                self.subcommands[self.arguments.command].run()
        self.call_plugins("on_end")