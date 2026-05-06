def sendEmail(self, emails, mass_type='SingleEmailMessage'):
        """
        Send one or more emails from Salesforce.

        Parameters:
            emails - a dictionary or list of dictionaries, each representing
                     a single email as described by https://www.salesforce.com
                     /us/developer/docs/api/Content/sforce_api_calls_sendemail
                     .htm
            massType - 'SingleEmailMessage' or 'MassEmailMessage'.
                       MassEmailMessage is used for mailmerge of up to 250
                       recepients in a single pass.

        Note:
            Newly created Salesforce Sandboxes default to System email only.
            In this situation, sendEmail() will fail with
            NO_MASS_MAIL_PERMISSION.
        """
        preparedEmails = _prepareSObjects(emails)
        if isinstance(preparedEmails, dict):
            # If root element is a dict, then this is a single object not an
            # array
            del preparedEmails['fieldsToNull']
        else:
            # else this is an array, and each elelment should be prepped.
            for listitems in preparedEmails:
                del listitems['fieldsToNull']
        res = BaseClient.sendEmail(self, preparedEmails, mass_type)
        if type(res) not in (TupleType, ListType):
            res = [res]
        data = list()
        for resu in res:
            d = dict()
            data.append(d)
            d['success'] = success = _bool(resu[_tPartnerNS.success])
            if not success:
                d['errors'] = [_extractError(e)
                               for e in resu[_tPartnerNS.errors,]]
            else:
                d['errors'] = list()
        return data