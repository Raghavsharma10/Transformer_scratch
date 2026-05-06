def _get_p_p_id_and_contract(self):
        """Get id of consumption profile."""
        contracts = {}
        try:
            raw_res = yield from self._session.get(PROFILE_URL,
                                                   timeout=self._timeout)
        except OSError:
            raise PyHydroQuebecError("Can not get profile page")
        # Parse html
        content = yield from raw_res.text()
        soup = BeautifulSoup(content, 'html.parser')
        # Search contracts
        for node in soup.find_all('span', {"class": "contrat"}):
            rematch = re.match("C[a-z]* ([0-9]{4} [0-9]{5})", node.text)
            if rematch is not None:
                contracts[rematch.group(1).replace(" ", "")] = None
        # search for links
        for node in soup.find_all('a', {"class": "big iconLink"}):
            for contract in contracts:
                if contract in node.attrs.get('href'):
                    contracts[contract] = node.attrs.get('href')
        # Looking for p_p_id
        p_p_id = None
        for node in soup.find_all('span'):
            node_id = node.attrs.get('id', "")
            if node_id.startswith("p_portraitConsommation_WAR"):
                p_p_id = node_id[2:]
                break

        if p_p_id is None:
            raise PyHydroQuebecError("Could not get p_p_id")

        return p_p_id, contracts