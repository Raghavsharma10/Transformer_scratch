def from_json(cls, json_string):
        """
        Creates and return a DataFrame from a JSON of the type created by to_json

        :param json_string: JSON
        :return: DataFrame
        """
        input_dict = json.loads(json_string)
        # convert index to tuple if required
        if input_dict['index'] and isinstance(input_dict['index'][0], list):
            input_dict['index'] = [tuple(x) for x in input_dict['index']]
        # convert index_name to tuple if required
        if isinstance(input_dict['meta_data']['index_name'], list):
            input_dict['meta_data']['index_name'] = tuple(input_dict['meta_data']['index_name'])
        data = input_dict['data'] if input_dict['data'] else None
        return cls(data=data, index=input_dict['index'], **input_dict['meta_data'])