def unarchive_user(self, user_id):
        """Unarchives the user with the specified user ID.

        Args:
            user_id: `int`. The ID of the user to unarchive.

        Returns:
            `NoneType`: None.
        """
        url = self.record_url + "/unarchive"
        res = requests.patch(url=url, json={"user_id": user_id}, headers=HEADERS, verify=False)
        self.write_response_html_to_file(res,"bob.html")
        res.raise_for_status()