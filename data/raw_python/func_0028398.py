def save_typed_params(self, data):
        """
        Send custom parameters descriptions to Polyglot to be used
        in front end UI configuration screen
        Accepts list of objects with the followin properties
            name - used as a key when data is sent from UI
            title - displayed in UI
            defaultValue - optionanl
            type - optional, can be 'NUMBER', 'STRING' or 'BOOLEAN'.
                Defaults to 'STRING'
            desc - optional, shown in tooltip in UI
            isRequired - optional, True/False, when set, will not validate UI
                input if it's empty
            isList - optional, True/False, if set this will be treated as list
                of values or objects by UI
            params - optional, can contain a list of objects. If present, then
                this (parent) is treated as object / list of objects by UI,
                otherwise, it's treated as a single / list of single values
        """
        LOGGER.info('Sending typed parameters to Polyglot.')
        if type(data) is not list:
            data = [ data ]
        message = { 'typedparams': data }
        self.send(message)