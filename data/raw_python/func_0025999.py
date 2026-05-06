def _butter(self, data, btype, f3=2, order=2):
        """
        Applies a digital butterworth filter via filtfilt at the specified f3 and order. Default values are set to 
        correspond to apparently sensible filters that distinguish between vibration and tilt from an accelerometer.
        :param data: the data to filter.
        :param btype: high or low.
        :param f3: the f3 of the filter.
        :param order: the filter order.
        :return: the filtered signal.
        """
        b, a = signal.butter(order, f3 / (0.5 * self.fs), btype=btype)
        y = signal.filtfilt(b, a, data)
        return y