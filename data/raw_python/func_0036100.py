def set_yaxis(self, param, unit=None, label=None):
        """ Sets the value of use on the yaxis
        :param param: value to use on the yaxis, should be a variable or function of the objects in objectList. ie 'R'
        for the radius variable and 'calcDensity()' for the calcDensity function

        :param unit: the unit to scale the values to
        :type unit: quantities unit or None

        :param label: axis label to use, if None "Parameter (Unit)" is generated here and used
        :type label: str
        """
        if unit is None:
            unit = self._getParLabelAndUnit(param)[1]  # use the default unit defined in this class
        self._yaxis_unit = unit

        self._yaxis = self._set_axis(param, unit)
        if label is None:
            self.ylabel = self._gen_label(param, unit)
        else:
            self.ylabel = label