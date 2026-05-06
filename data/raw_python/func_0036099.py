def set_xaxis(self, param, unit=None, label=None):
        """ Sets the value of use on the x axis
        :param param: value to use on the xaxis, should be a variable or function of the objects in objectList. ie 'R'
        for the radius variable and 'calcDensity()' for the calcDensity function

        :param unit: the unit to scale the values to, None will use the default
        :type unit: quantities unit or None

        :param label: axis label to use, if None "Parameter (Unit)" is generated here and used
        :type label: str
        """

        if unit is None:
            unit = self._getParLabelAndUnit(param)[1]  # use the default unit defined in this class
        self._xaxis_unit = unit

        self._xaxis = self._set_axis(param, unit)
        if label is None:
            self.xlabel = self._gen_label(param, unit)
        else:
            self.xlabel = label