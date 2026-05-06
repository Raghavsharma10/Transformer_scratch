def write(self):
        """ Write configuration to chassis.

        Raise StreamWarningsError if configuration warnings found.
        """

        self.ix_command('write')
        stream_warnings = self.streamRegion.generateWarningList()
        warnings_list = (self.api.call('join ' + ' {' + stream_warnings + '} ' + ' LiStSeP').split('LiStSeP')
                         if self.streamRegion.generateWarningList() else [])
        for warning in warnings_list:
            if warning:
                raise StreamWarningsError(warning)