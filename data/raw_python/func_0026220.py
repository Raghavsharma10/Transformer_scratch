def _load_contract_page(self, contract_url):
        """Load the profile page of a specific contract when we have multiple contracts."""
        try:
            yield from self._session.get(contract_url,
                                         timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecError("Can not get profile page for a "
                                     "specific contract")