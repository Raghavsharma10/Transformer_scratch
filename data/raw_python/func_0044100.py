def ingress_filter(self, response):
        """ Flatten a response with meta and data keys into an object. """
        data = self.data_getter(response)
        if isinstance(data, dict):
            data = m_data.DictResponse(data)
        elif isinstance(data, list):
            data = m_data.ListResponse(data)
        else:
            return data
        data.meta = self.meta_getter(response)
        return data