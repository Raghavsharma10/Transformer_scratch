def insert(self, filename):
        """
        Parses files to load them into memory and insert them into the class.

        :param filename: File or directory pointing to .ioc files.
        :return: A list of .ioc files which could not be parsed.
        """
        errors = []
        if os.path.isfile(filename):
            log.info('loading IOC from: {}'.format(filename))
            try:
                self.parse(ioc_api.IOC(filename))
            except ioc_api.IOCParseError:
                log.exception('Parse Error')
                errors.append(filename)
        elif os.path.isdir(filename):
            log.info('loading IOCs from: {}'.format(filename))
            for fn in glob.glob(filename + os.path.sep + '*.ioc'):
                if not os.path.isfile(fn):
                    continue
                else:
                    try:
                        self.parse(ioc_api.IOC(fn))
                    except ioc_api.IOCParseError:
                        log.exception('Parse Error')
                        errors.append(fn)
        else:
            pass
        log.info('Parsed [{}] IOCs'.format(len(self)))
        return errors