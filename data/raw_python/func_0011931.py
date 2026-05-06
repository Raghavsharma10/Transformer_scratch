def _generate_request(self, callname, request):
        """Generate a request object for delivery to the API"""
        # Retrieve path from API class
        schema = self.api.request_schema()
        schema.context['callname'] = callname
        return schema.dump(request).data.get("payload")