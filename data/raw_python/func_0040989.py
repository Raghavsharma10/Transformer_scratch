def account(self):
        """
        Get the API account details like balance of credits.
        :return: An Account object.
        """
        resp = self._call(endpoint='account')
        return Account(resp['credits'], resp['jobs_completed'], resp['jobs_processing'])