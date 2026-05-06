def get_mnemonics (self, mnemonics):
        """The method get series by mnemonics"""
        path = '/api/1.0/data/mnemonics?mnemonics={0}'
        return self._api_get(definition.MnemonicsResponseList, path.format(mnemonics))