async def post(self):
        """
        Accepts json-rpc post request.
        Retrieves data from request body.
        Calls defined method in field 'method_name'
        """

        request = self.request.body.decode()
        response = await methods.dispatch(request)
        if not response.is_notification:
            self.write(response)