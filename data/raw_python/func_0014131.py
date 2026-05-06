def get_updated_property_details(self, zpid):
        """
        GetUpdatedPropertyDetails API
        """
        url = 'http://www.zillow.com/webservice/GetUpdatedPropertyDetails.htm'

        params = {
            'zpid': zpid,
            'zws-id': self.api_key
        }
        return self.get_data(url, params)