def _get_balances(self):
        """Get all balances.

        .. todo::

            IT SEEMS balances are shown (MAIN_URL) in the same order
            that contracts in profile page (PROFILE_URL).
            Maybe we should ensure that.
        """
        balances = []
        try:
            raw_res = yield from self._session.get(MAIN_URL,
                                                   timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecError("Can not get main page")
        # Parse html
        content = yield from raw_res.text()
        soup = BeautifulSoup(content, 'html.parser')
        solde_nodes = soup.find_all("div", {"class": "solde-compte"})
        if solde_nodes == []:
            raise PyHydroQuebecError("Can not found balance")
        for solde_node in solde_nodes:
            try:
                balance = solde_node.find("p").text
            except AttributeError:
                raise PyHydroQuebecError("Can not found balance")
            balances.append(float(balance[:-2]
                            .replace(",", ".")
                            .replace("\xa0", "")))

        return balances