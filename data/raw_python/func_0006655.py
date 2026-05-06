def get_options(self, section='default', opt_keys=None, vars=None):
        """
        Get all options for a section.  If ``opt_keys`` is given return
        only options with those keys.
        """
        vars = vars if vars else self.default_vars
        conf = self.parser
        opts = {}
        if opt_keys is None:
            if conf is None:
                opt_keys = {}
            else:
                if not self.robust or conf.has_section(section):
                    opt_keys = conf.options(section)
                else:
                    opt_keys = {}
        else:
            logger.debug('conf: %s' % conf)
            copts = conf.options(section) if conf else {}
            opt_keys = set(opt_keys).intersection(set(copts))
        for option in opt_keys:
            logger.debug(f'option: {option}, vars: {vars}')
            opts[option] = conf.get(section, option, vars=vars)
        return opts