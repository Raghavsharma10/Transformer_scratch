def params(self):
        """Parameters used in the url of the API call and for authentication.

        :return: parameters used in the url.
        :rtype: dict
        """
        params = {}
        params["access_token"] = self.access_token
        params["account_id"] = self.account_id
        params["alert_id"] = self.alert_id

        if self.since_id:
            params["since_id"] = self.since_id
        else:
            params["before_date"] = self.before_date if self.before_date else ""
            params["not_before_date"] = self.not_before_date if self.before_date else ""
            params["cursor"] = self.cursor if self.cursor else ""

        if self.unread:
            params["unread"] = self.unread
        else:
            if (self.favorite) and (
                (self.folder == "inbox") or (self.folder == "archive")):
                params["favorite"] = self.favorite
                params["folder"] = self.folder
            else:
                 params["folder"] = self.folder if self.folder else ""   
            params["q"] = self.q if self.q else ""
            params["tone"] = self.tone if self.tone else ""

        if int(self.limit) > 1000:
            params["limit"] = "1000"
        elif int(self.limit) < 1:
            params["limit"] = ""
        else:
            params["limit"] = self.limit

        params["source"] = self.source if self.source else ""

        params["countries"] = self.countries if self.countries else ""
        params["include_children"] = self.include_children if self.include_children else ""
        params["sort"] = self.sort if self.sort else ""
        params["languages"] = self.languages if self.languages else ""
        params["timezone"] = self.timezone if self.timezone else ""

        # Deletes parameter if it does not have a value
        for key, value in list(params.items()):
            if value == '':
                del params[key]

        return params