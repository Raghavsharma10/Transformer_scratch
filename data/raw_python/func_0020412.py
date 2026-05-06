def get_consumption(self, deviceid, timerange="10"):
        """
        Return all available energy consumption data for the device.
        You need to divice watt_values by 100 and volt_values by 1000
        to get the "real" values.

        :return: dict
        """
        tranges = ("10", "24h", "month", "year")
        if timerange not in tranges:
            raise ValueError(
                "Unknown timerange. Possible values are: {0}".format(tranges)
            )

        url = self.base_url + "/net/home_auto_query.lua"
        response = self.session.get(url, params={
            'sid': self.sid,
            'command': 'EnergyStats_{0}'.format(timerange),
            'id': deviceid,
            'xhr': 0,
        }, timeout=15)
        response.raise_for_status()

        data = response.json()
        result = {}

        # Single result values
        values_map = {
            'MM_Value_Amp': 'mm_value_amp',
            'MM_Value_Power': 'mm_value_power',
            'MM_Value_Volt': 'mm_value_volt',

            'EnStats_average_value': 'enstats_average_value',
            'EnStats_max_value': 'enstats_max_value',
            'EnStats_min_value': 'enstats_min_value',
            'EnStats_timer_type': 'enstats_timer_type',

            'sum_Day': 'sum_day',
            'sum_Month': 'sum_month',
            'sum_Year': 'sum_year',
        }
        for avm_key, py_key in values_map.items():
            result[py_key] = int(data[avm_key])

        # Stats counts
        count = int(data["EnStats_count"])
        watt_values = [None for i in range(count)]
        volt_values = [None for i in range(count)]
        for i in range(1, count + 1):
            watt_values[i - 1] = int(data["EnStats_watt_value_{}".format(i)])
            volt_values[i - 1] = int(data["EnStats_volt_value_{}".format(i)])

        result['watt_values'] = watt_values
        result['volt_values'] = volt_values

        return result