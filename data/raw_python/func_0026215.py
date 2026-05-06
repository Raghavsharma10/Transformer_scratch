def _get_login_page(self):
        """Go to the login page."""
        try:
            raw_res = yield from self._session.get(HOME_URL,
                                                   timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecError("Can not connect to login page")
        # Get login url
        content = yield from raw_res.text()
        soup = BeautifulSoup(content, 'html.parser')
        form_node = soup.find('form', {'name': 'fm'})
        if form_node is None:
            raise PyHydroQuebecError("No login form find")
        login_url = form_node.attrs.get('action')
        if login_url is None:
            raise PyHydroQuebecError("Can not found login url")
        return login_url