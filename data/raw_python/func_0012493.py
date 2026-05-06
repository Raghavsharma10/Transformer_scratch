def _limit_data(self, data):
        """
        Find the per-day average of each series in the data over the last 7
        days; drop all but the top 10.

        :param data: original graph data
        :type data: dict
        :return: dict containing only the top 10 series, based on average over
          the last 7 days.
        :rtype: dict
        """
        if len(data.keys()) <= 10:
            logger.debug("Data has less than 10 keys; not limiting")
            return data
        # average last 7 days of each series
        avgs = {}
        for k in data:
            if len(data[k]) <= 7:
                vals = data[k]
            else:
                vals = data[k][-7:]
            avgs[k] = sum(vals) / len(vals)
        # hold state
        final_data = {}  # final data dict
        other = []  # values for dropped/'other' series
        count = 0  # iteration counter
        # iterate the sorted averages; either drop or keep
        for k in sorted(avgs, key=avgs.get, reverse=True):
            if count < 10:
                final_data[k] = data[k]
                logger.debug("Keeping data series %s (average over last 7 "
                             "days of data: %d", k, avgs[k])
            else:
                logger.debug("Adding data series %s to 'other' (average over "
                             "last 7 days of data: %d", k, avgs[k])
                other.append(data[k])
            count += 1
        # sum up the other data and add to final
        final_data['other'] = [sum(series) for series in zip(*other)]
        return final_data