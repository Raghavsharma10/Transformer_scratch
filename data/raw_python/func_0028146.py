def _get_credentials(self):
        # type: () -> Optional[Tuple[str, str]]
        """
        Return HDX site username and password

        Returns:
            Optional[Tuple[str, str]]: HDX site username and password or None

        """
        site = self.data[self.hdx_site]
        username = site.get('username')
        if username:
            return b64decode(username).decode('utf-8'), b64decode(site['password']).decode('utf-8')
        else:
            return None