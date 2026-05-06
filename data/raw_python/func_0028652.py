def _reformat_matrix(self):
        """Transform a square matrix into a format with two independent variables and one dependent variable.
        """
        nxax = len(self.current_table.data['independent_variables'])
        nyax = len(self.current_table.data['dependent_variables'])
        npts = len(self.current_table.data['dependent_variables'][0]['values'])

        # check if 1 x-axis, and npts (>=2) equals number of y-axes
        if nxax != 1 or nyax != npts or npts < 2:
            return False

        # add second independent variable with each value duplicated npts times
        if len(self.current_table.xheaders) == 2:
            xheader = self.current_table.xheaders[1]
        else:
            xheader = copy.deepcopy(self.current_table.data['independent_variables'][0]['header'])
        self.current_table.data['independent_variables'].append({'header': xheader, 'values': []})
        for value in self.current_table.data['independent_variables'][0]['values']:
            self.current_table.data['independent_variables'][1]['values'].extend([copy.deepcopy(value) for npt in range(npts)])

        # duplicate values of first independent variable npts times
        self.current_table.data['independent_variables'][0]['values'] \
            = [copy.deepcopy(value) for npt in range(npts) for value in self.current_table.data['independent_variables'][0]['values']]

        # suppress header if different for second y-axis
        if self.current_table.data['dependent_variables'][0]['header'] != \
                self.current_table.data['dependent_variables'][1]['header']:
            self.current_table.data['dependent_variables'][0]['header'] = {'name': ''}

        # remove qualifier if different for second y-axis
        iqdel = [] # list of qualifier indices to be deleted
        for iq, qualifier in enumerate(self.current_table.data['dependent_variables'][0]['qualifiers']):
            if qualifier != self.current_table.data['dependent_variables'][1]['qualifiers'][iq]:
                iqdel.append(iq)
        for iq in iqdel[::-1]: # need to delete in reverse order
            del self.current_table.data['dependent_variables'][0]['qualifiers'][iq]

        # append values of second and subsequent y-axes to first dependent variable
        for iy in range(1, nyax):
            for value in self.current_table.data['dependent_variables'][iy]['values']:
                self.current_table.data['dependent_variables'][0]['values'].append(value)

        # finally, delete the second and subsequent y-axes in reverse order
        for iy in range(nyax-1, 0, -1):
            del self.current_table.data['dependent_variables'][iy]

        return True