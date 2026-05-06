def to_json(self):
        """
        Returns a JSON of the entire DataFrame that can be reconstructed back with raccoon.from_json(input). Any object
        that cannot be serialized will be replaced with the representation of the object using repr(). In that instance
        the DataFrame will have a string representation in place of the object and will not reconstruct exactly.

        :return: json string
        """
        input_dict = {'data': self.to_dict(index=False), 'index': list(self._index)}

        # if blist, turn into lists
        if self.blist:
            input_dict['index'] = list(input_dict['index'])
            for key in input_dict['data']:
                input_dict['data'][key] = list(input_dict['data'][key])

        meta_data = dict()
        for key in self.__slots__:
            if key not in ['_data', '_index']:
                value = self.__getattribute__(key)
                meta_data[key.lstrip('_')] = value if not isinstance(value, blist) else list(value)
        meta_data['use_blist'] = meta_data.pop('blist')
        input_dict['meta_data'] = meta_data
        return json.dumps(input_dict, default=repr)