def _check_response_for_request_errors(self):
        """
        Checks the response to see if there were any errors specific to
        this WSDL.
        """
        if self.response.HighestSeverity == "ERROR":
            for notification in self.response.Notifications:  # pragma: no cover
                if notification.Severity == "ERROR":
                    if "Postal Code Not Found" in notification.Message:
                        raise FedexPostalCodeNotFound(notification.Code,
                                                      notification.Message)

                    elif "Invalid Postal Code Format" in self.response.Notifications:
                        raise FedexInvalidPostalCodeFormat(notification.Code,
                                                           notification.Message)
                    else:
                        raise FedexError(notification.Code,
                                         notification.Message)