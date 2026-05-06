def _process_response(self):
        """Return a JSON result after an HTTP Request.

        Process the response of an HTTP Request and make it a JSON error if
        it failed. Otherwise return the response's content.

        """
        response = self.conn.getresponse()
        if response.status == 200 or response.status == 201:
            data = response.read()
        else:
            data = {
                "error":  {
                    "code": response.status,
                    "reason": response.reason,
                    "data": response.read()
                }
            }

        return data