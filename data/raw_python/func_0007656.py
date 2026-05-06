def profiles(self):
        """
        A list of all profiles on this web property. You may
        select a specific profile using its name, its id
        or an index.

        ```python
        property.profiles[0]
        property.profiles['9234823']
        property.profiles['marketing profile']
        ```
        """
        raw_profiles = self.account.service.management().profiles().list(
            accountId=self.account.id,
            webPropertyId=self.id).execute()['items']
        profiles = [Profile(raw, self) for raw in raw_profiles]
        return addressable.List(profiles, indices=['id', 'name'], insensitive=True)