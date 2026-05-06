def insert(self, filename):
        """
        Parses files to load them into memory and insert them into the class.

        :param filename: File or directory pointing to .ioc files.
        :return: A list of .ioc files which could not be parsed.
        """
        errors = []
        if os.path.isfile(filename):
            log.info('loading IOC from: {}'.format(filename))
            if not self.parse(filename):
                log.warning('Failed to prase [{}]'.format(filename))
                errors.append(filename)
        elif os.path.isdir(filename):
            log.info('loading IOCs from: {}'.format(filename))
            for fn in glob.glob(filename + os.path.sep + '*.ioc'):
                if not os.path.isfile(fn):
                    continue
                else:
                    if not self.parse(fn):
                        log.warning('Failed to parse [{}]'.format(filename))
                        errors.append(fn)
        else:
            pass
        log.info('Parsed [%s] IOCs' % str(len(self)))
        return errors