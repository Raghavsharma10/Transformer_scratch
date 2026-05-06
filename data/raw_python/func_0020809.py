def requestTimingInfo(self):
        """
        Returns the time needed to process the request by the frontend server in microseconds
        and the EPOC timestamp of the request in microseconds.

        :rtype: tuple containing processing time and timestamp
        """
        try:
            return tuple(item.split('=')[1] for item in self.http_response.header.get('CMS-Server-Time').split())
        except AttributeError:
            return None, None