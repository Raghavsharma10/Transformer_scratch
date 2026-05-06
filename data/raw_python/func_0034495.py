def load(self):
        """Load our config, log and raise on error."""
        try:
            merged_configfile = self.get_merged_config()
            self.yamldocs = yaml.load(merged_configfile, Loader=Loader)

            # Strip out the top level 'None's we get from concatenation.
            # Functionally not required, but makes dumps cleaner.
            self.yamldocs = [x for x in self.yamldocs if x]
            self.logdebug('parsed_rules:\n%s\n' % pretty(self.yamldocs))

        except (yaml.scanner.ScannerError, yaml.parser.ParserError):
            self.raise_and_log_error(ConfigError, 'error parsing config.')