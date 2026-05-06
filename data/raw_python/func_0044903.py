def list_emails(self, page=0, since=None):
        """
        Get the emails of current object

        :param page: the page starting at 0
        :type since: int
        :param since: get all notes since a datetime
        :type since: datetime.datetime
        :return: the emails
        :rtype: list
        """
        from highton.models.email import Email
        params = {'n': int(page) * self.EMAILS_OFFSET}

        if since:
            params['since'] = since.strftime(self.COLLECTION_DATETIME)

        return fields.ListField(
            name=self.ENDPOINT,
            init_class=Email
        ).decode(
            self.element_from_string(
                self._get_request(
                    endpoint=self.ENDPOINT + '/' + str(self.id) + '/' + Email.ENDPOINT,
                    params=params
                ).text
            )
        )