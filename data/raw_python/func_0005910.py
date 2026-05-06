def format(self, do_print=False, stamp=True):
        """Applies formatting to configuration.

        *Currently formats to .ini*

        :param bool do_print: Whether to print out formatted config.
        :param bool stamp: Whether to add stamp data to the first configuration section.
        :rtype: str|unicode
        """
        if stamp and self.sections:
            self.sections[0].print_stamp()

        formatted = IniFormatter(self.sections).format()

        if do_print:
            print(formatted)

        return formatted