def _get_inputs(self):
        """Converts command line args into a list of template inputs
        """
        # Convert file inputs to a dict, to make it easier to override
        # them with commandline inputs
        file_inputs = self._get_file_inputs()
        try:
            jsonschema.validate(file_inputs, file_input_schema)
        except jsonschema.ValidationError:
            raise SystemExit("ERROR! Input file was invalid")
        input_dict = {}
        for (channel, input_id) in file_inputs.iteritems():
            input_dict[channel] = input_id

        if self.args.inputs:
            for kv_pair in self.args.inputs:
                (channel, input_id) = kv_pair.split('=')
                input_dict[channel] = self._parse_string_to_nested_lists(
                    input_id)

        inputs = []
        for (channel, contents) in input_dict.iteritems():
            inputs.append({
                'channel': channel,
                'data': {
                    'contents': contents
                }
            })
        return inputs