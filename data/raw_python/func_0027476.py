def modifyContacts(self, contactids, paused):
        """Modifies a list of contacts.

        Provide comma separated list of contact ids and desired paused state

        Returns status message
        """

        response = self.request("PUT", "notification_contacts", {'contactids': contactids,
                                                    'paused': paused})
        return response.json()['message']