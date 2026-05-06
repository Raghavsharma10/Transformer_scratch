def sendEmail(self, emails, massType='SingleEmailMessage'):
        """
        Send one or more emails from Salesforce.

        Parameters:
            emails - a dictionary or list of dictionaries, each representing a
                     single email as described by https://www.salesforce.com/us
                     /developer/docs/api/Content/sforce_api_calls_sendemail.htm
            massType - 'SingleEmailMessage' or 'MassEmailMessage'.
                       MassEmailMessage is used for mailmerge of up to 250
                       recepients in a single pass.

        Note:
            Newly created Salesforce Sandboxes default to System email only. In
            this situation, sendEmail() will fail with NO_MASS_MAIL_PERMISSION.
        """
        return SendEmailRequest(
            self.__serverUrl,
            self.sessionId,
            emails,
            massType
        ).post(self.__conn)