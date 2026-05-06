def parser(self):
        "Load the configuration file."
        if not hasattr(self, '_conf'):
            cfile = self.config_file
            logger.debug('loading config %s' % cfile)
            if os.path.isfile(cfile):
                conf = self._create_config_parser()
                conf.read(os.path.expanduser(cfile))
            else:
                if self.robust:
                    logger.debug('no default config file %s--skipping' % cfile)
                else:
                    raise IOError('no such file: %s' % cfile)
                conf = None
            self._conf = conf
        return self._conf