def _check_response_for_request_errors(self):
        """
        Override this in each service module to check for errors that are
        specific to that module. For example, invalid tracking numbers in
        a Tracking request.
        """

        if self.response.HighestSeverity == "ERROR":
            for notification in self.response.Notifications:
                if notification.Severity == "ERROR":
                    raise FedexError(notification.Code,
                                     notification.Message)