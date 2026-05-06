def import_contacts(self, email, password, include_name=False):
        """
        Fetch email contacts from a user's address book on one of the major email websites. Currently supports AOL, Gmail, Hotmail, and Yahoo! Mail.
        """
        data = {'email': email,
                'password': password}
        if include_name:
            data['names'] = 1
        return self.api_post('contacts', data)