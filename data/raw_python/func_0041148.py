def params(self):
        """Parameters used in the url of the API call and for authentication.

        :return: parameters used in the url.
        :rtype: dict
        """
        params = {}
        params["access_token"] = self.access_token
        params["account_id"] = self.account_id
        params["alert_id"] = self.alert_id
        params["mention_id"] = self.mention_id
        params["before_date"] = self.before_date if self.before_date else ""

        if self.limit:
            if int(self.limit) > 1000:
                params["limit"] = "1000"
            elif int(self.limit) < 1:
                params["limit"] = ""
            else:
                params["limit"] = self.limit

        return params