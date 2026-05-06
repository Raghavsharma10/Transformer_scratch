def config_args(self, section='default'):
        """Loop through the configuration file and set all of our values.

        Note:
          that anything can be set as a "section" in the argument file. If a
          section does not exist an empty dict will be returned.


        :param section: ``str``
        :return: ``dict``
        """
        if sys.version_info >= (2, 7, 0):
            parser = ConfigParser.SafeConfigParser(allow_no_value=True)
        else:
            parser = ConfigParser.SafeConfigParser()

        # Set to preserve Case
        parser.optionxform = str
        args = {}
        try:
            parser.read(self.config_file)
            for name, value in parser.items(section):
                if any([value == 'False', value == 'false']):
                    value = False
                elif any([value == 'True', value == 'true']):
                    value = True
                else:
                    value = utils.is_int(value=value)
                args[name] = value
        except Exception as exp:
            self.log.warn('Section: [ %s ] Message: "%s"', section, exp)
            return {}
        else:
            return args