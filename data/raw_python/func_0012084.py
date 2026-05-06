def _generate_result(self, res_type, channel, result):
        """Generate the result object"""
        schema = self.api.ws_result_schema()
        schema.context['channel'] = channel
        schema.context['response_type'] = res_type
        self.callback(schema.load(result), self.context)