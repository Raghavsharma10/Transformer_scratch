def webproperties(self):
        """
        A list of all web properties on this account. You may
        select a specific web property using its name, its id
        or an index.

        ```python
        account.webproperties[0]
        account.webproperties['UA-9234823-5']
        account.webproperties['debrouwere.org']
        ```
        """

        raw_properties = self.service.management().webproperties().list(
            accountId=self.id).execute()['items']
        _webproperties = [WebProperty(raw, self) for raw in raw_properties]
        return addressable.List(_webproperties, indices=['id', 'name'], insensitive=True)