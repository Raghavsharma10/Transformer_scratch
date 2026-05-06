def peak_templates(self):
        """Create a list of concrete peak templates from a list of general peak descriptions.

        :return: List of peak templates.
        :rtype: :py:class:`list`
        """
        peak_templates = []
        for peak_descr in self:
            expanded_dims = [dim_group.dimensions for dim_group in peak_descr]
            templates = product(*expanded_dims)
            for template in templates:
                peak_templates.append(PeakTemplate(template))
        return peak_templates