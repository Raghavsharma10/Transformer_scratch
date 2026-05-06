def _get_inputs(self, old_inputs):
        """Converts command line args into a list of template inputs
        """
        # Convert inputs to dict to facilitate overriding by channel name
        # Also, drop DataNode ID and keep only contents.
        input_dict = {}
        for input in old_inputs:
            # Strip out DataNode UUID and URL
            input['data'] = {'contents': input['data']['contents']}
            input_dict[input['channel']] = input

        file_inputs = self._get_file_inputs()
        try:
            jsonschema.validate(file_inputs, file_input_schema)
        except jsonschema.ValidationError:
            raise SystemExit("ERROR! User inputs file is not valid")
        for (channel, input_id) in file_inputs.iteritems():
            input_dict[channel] = {
                'channel': channel,
                'data': {'contents': input_id}
            }
        # Override with cli user inputs if specified
        if self.args.inputs:
            for kv_pair in self.args.inputs:
                (channel, input_id) = kv_pair.split('=')
                input_dict[channel] = {
                    'channel': channel,
                    'data': {
                        'contents':
                        self._parse_string_to_nested_lists(input_id)}
                }
        return input_dict.values()