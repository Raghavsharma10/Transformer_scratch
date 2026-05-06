def get_xchange_rate(self, pairs, items=None):
        """Retrieves currency exchange rate data for given pair(s). 
        Accepts both where pair='eurusd, gbpusd' and where pair in ('eurusd', 'gpbusd, usdaud')
        """
        response = self.select('yahoo.finance.xchange', items).where(['pair', 'in', pairs])
        return response