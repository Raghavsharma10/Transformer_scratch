def _parse(self, content):
        """
            Parse data request to data from python.

            @param content: Context of request.

            @raise ParseError:
        """
        if content:

            stream = BytesIO(str(content))
            data = json.loads(stream.getvalue())

            return data