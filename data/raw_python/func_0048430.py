def _report_error(self, request, exp):
        """When making the request, if an error happens, log it."""
        message = (
            "Failure to perform %s due to [ %s ]" % (request, exp)
        )
        self.log.fatal(message)
        raise requests.RequestException(message)