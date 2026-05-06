def fetch_data_detailled_energy_use(self, start_date=None, end_date=None):
        """Get detailled energy use from a specific contract."""
        if start_date is None:
            start_date = datetime.datetime.now(HQ_TIMEZONE) - datetime.timedelta(days=1)
        if end_date is None:
            end_date = datetime.datetime.now(HQ_TIMEZONE)
        # Get http session
        yield from self._get_httpsession()
        # Get login page
        login_url = yield from self._get_login_page()
        # Post login page
        yield from self._post_login_page(login_url)
        # Get p_p_id and contracts
        p_p_id, contracts = yield from self._get_p_p_id_and_contract()
        # If we don't have any contrats that means we have only
        # onecontract. Let's get it
        if contracts == {}:
            contracts = yield from self._get_lonely_contract()
        # For all contracts
        for contract, contract_url in contracts.items():
            if contract_url:
                yield from self._load_contract_page(contract_url)

            data = {}
            dates = [(start_date + datetime.timedelta(n))
                     for n in range(int((end_date - start_date).days))]

            for date in dates:
                # Get Hourly data
                day_date = date.strftime("%Y-%m-%d")
                hourly_data = yield from self._get_hourly_data(day_date, p_p_id)
                data[day_date] = hourly_data['raw_hourly_data']

            # Add contract
            self._data[contract] = data