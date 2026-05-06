def set_codes(self, codes):
        '''Set the country code map for the data.
        Codes given in a list.

        i.e. DE - Germany
             AT - Austria
             US - United States
        '''

        codemap = ''
        
        for cc in codes:
            cc = cc.upper()
            if cc in self.__ccodes:
                codemap += cc
            else:
                raise UnknownCountryCodeException(cc)
            
        self.codes = codemap