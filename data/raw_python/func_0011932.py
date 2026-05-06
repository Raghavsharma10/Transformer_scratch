def _generate_result(self, callname, result):
        """Generate a results object for delivery to the context object"""
        # Retrieve path from API class
        schema = self.api.result_schema()
        schema.context['callname'] = callname
        self.callback(schema.load(result), self.context)