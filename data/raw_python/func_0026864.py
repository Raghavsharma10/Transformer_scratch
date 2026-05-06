def get_raw_data(self, times=5):
        """
        do some readings and aggregate them using the defined statistics function

        :param times: how many measures to aggregate
        :type times: int
        :return: the aggregate of the measured values
        :rtype float
        """

        self._validate_measure_count(times)

        data_list = []
        while len(data_list) < times:
            data = self._read()
            if data not in [False, -1]:
                data_list.append(data)

        return data_list