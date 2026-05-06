def _check_response_for_request_warnings(self):
        """
        Override this in a service module to check for errors that are
        specific to that module. For example, changing state/province based
        on postal code in a Rate Service request.
        """

        if self.response.HighestSeverity in ("NOTE", "WARNING"):
            for notification in self.response.Notifications:
                if notification.Severity in ("NOTE", "WARNING"):
                    self.logger.warning(FedexFailure(notification.Code,
                                                     notification.Message))