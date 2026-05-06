def __check_response_for_fedex_error(self):
        """
        This checks the response for general Fedex errors that aren't related
        to any one WSDL.
        """

        if self.response.HighestSeverity == "FAILURE":
            for notification in self.response.Notifications:
                if notification.Severity == "FAILURE":
                    raise FedexFailure(notification.Code,
                                       notification.Message)