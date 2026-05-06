def entries(self):
        """
        Provides access to entry management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/entries

        :return: :class:`EnvironmentEntriesProxy <contentful_management.environment_entries_proxy.EnvironmentEntriesProxy>` object.
        :rtype: contentful.environment_entries_proxy.EnvironmentEntriesProxy

        Usage:

            >>> environment_entries_proxy = environment.entries()
            <EnvironmentEntriesProxy space_id="cfexampleapi" environment_id="master">
        """

        return EnvironmentEntriesProxy(self._client, self.space.id, self.id)