def fetch_data(self):
        """Get the latest data from HydroQuebec."""
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

        # Get balance
        balances = yield from self._get_balances()
        balances_len = len(balances)
        balance_id = 0
        # For all contracts
        for contract, contract_url in contracts.items():
            if contract_url:
                yield from self._load_contract_page(contract_url)

            # Get Hourly data
            try:
                yesterday = datetime.datetime.now(HQ_TIMEZONE) - datetime.timedelta(days=1)
                day_date = yesterday.strftime("%Y-%m-%d")
                hourly_data = yield from self._get_hourly_data(day_date, p_p_id)
                hourly_data = hourly_data['processed_hourly_data']
            except Exception:  # pylint: disable=W0703
                # We don't have hourly data for some reason
                hourly_data = {}

            # Get Annual data
            try:
                annual_data = yield from self._get_annual_data(p_p_id)
            except PyHydroQuebecAnnualError:
                # We don't have annual data, which is possible if your
                # contract is younger than 1 year
                annual_data = {}
            # Get Monthly data
            monthly_data = yield from self._get_monthly_data(p_p_id)
            monthly_data = monthly_data[0]
            # Get daily data
            start_date = monthly_data.get('dateDebutPeriode')
            end_date = monthly_data.get('dateFinPeriode')
            try:
                daily_data = yield from self._get_daily_data(p_p_id, start_date, end_date)
            except Exception:  # pylint: disable=W0703
                daily_data = []
            # We have to test daily_data because it's empty
            # At the end/starts of a period
            if daily_data:
                daily_data = daily_data[0]['courant']
            # format data
            contract_data = {"balance": balances[balance_id]}
            for key1, key2 in MONTHLY_MAP:
                contract_data[key1] = monthly_data[key2]
            for key1, key2 in ANNUAL_MAP:
                contract_data[key1] = annual_data.get(key2, "")
            # We have to test daily_data because it's empty
            # At the end/starts of a period
            if daily_data:
                for key1, key2 in DAILY_MAP:
                    contract_data[key1] = daily_data[key2]
            # Hourly
            if hourly_data:
                contract_data['yesterday_hourly_consumption'] = hourly_data
            # Add contract
            self._data[contract] = contract_data
            balance_count = balance_id + 1
            if balance_count < balances_len:
                balance_id += 1