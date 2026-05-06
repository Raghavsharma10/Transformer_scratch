def specstring(self):
        """The string corresponding to the current values of `subgroup`,
        `state`, and `variable`.

        >>> from hydpy.core.itemtools import ExchangeSpecification
        >>> spec = ExchangeSpecification('hland_v1', 'fluxes.qt')
        >>> spec.specstring
        'fluxes.qt'
        >>> spec.series = True
        >>> spec.specstring
        'fluxes.qt.series'
        >>> spec.subgroup = None
        >>> spec.specstring
        'qt.series'
        """
        if self.subgroup is None:
            variable = self.variable
        else:
            variable = f'{self.subgroup}.{self.variable}'
        if self.series:
            variable = f'{variable}.series'
        return variable