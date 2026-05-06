def add_data_dict(self, datadict):
        '''Sets the data and country codes via a dictionary.

        i.e. {'DE': 50, 'GB': 30, 'AT': 70}
        '''

        self.set_codes(list(datadict.keys()))
        self.add_data(list(datadict.values()))