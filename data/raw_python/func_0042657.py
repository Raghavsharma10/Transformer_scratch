def list_observatories(self):
        """
        Get the IDs of all observatories with have stored observations on this server.

        :return: a sequence of strings containing observatories IDs
        """
        response = requests.get(self.base_url + '/obstories').text
        return safe_load(response)