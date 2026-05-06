def get_response(self, data_type=None):
        """return json response from APIs converted into python list
           : param string 'data_type', if it's None it'll return the avaliable cache,
            if we've both global and ticker data, the function will return 'ticker' data,
            in that case, data_type should be assigned with 'ticker' or 'global'
        """
        if not data_type:
            return self.cache.get_response(r_type='ticker') or self.cache.get_response(r_type='global')
        elif data_type == 'ticker':
            return self.cache.get_response(r_type='ticker')
        return self.cache.get_response(r_type='global')