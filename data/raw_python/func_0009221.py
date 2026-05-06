def is_site_available(self):
        """
        Returns true if we can access LendingClub.com
        This is also a simple test to see if there's a network connection

        Returns
        -------
        boolean
            True or False
        """
        try:
            response = requests.head(self.base_url)
            status = response.status_code
            return 200 <= status < 400  # Returns true if the status code is greater than 200 and less than 400
        except Exception:
            return False