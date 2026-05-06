def _set_axis(self, param, unit):
        """ this should take a variable or a function and turn it into a list by evaluating on each planet
        """
        axisValues = []
        for astroObject in self.objectList:
            try:
                value = eval('astroObject.{0}'.format(param))
            except ac.HierarchyError:  # ie trying to call planet.star and one planet is a lone ranger
                value = np.nan

            if unit is None:  # no unit to rescale (a aq.unitless quanitity would otherwise fail with ValueError)
                axisValues.append(value)
            else:
                try:
                    axisValues.append(value.rescale(unit))
                except AttributeError:  # either nan or unitless
                    axisValues.append(value)

        return axisValues