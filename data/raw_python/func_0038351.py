def output_spectrum(self, spectrum, filepath, header={}):
        """
        Prints a file of the given spectrum to an ascii file with specified filepath.

        Parameters
        ----------
        spectrum: int, sequence
            The id from the SPECTRA table or a [w,f,e] sequence
        filepath: str
            The path of the file to print the data to.
        header: dict
                A dictionary of metadata to add of update in the header

        """
        # If an integer is supplied, get the spectrum from the SPECTRA table
        if isinstance(spectrum, int):
            data = self.query("SELECT * FROM spectra WHERE id={}".format(spectrum), fetch='one', fmt='dict')
            try:
                data['header'] = list(map(list, data['spectrum'].header.cards)) + [[k, v, ''] for k, v in
                                                                                   header.items()]
            except:
                data['header'] = ''

        # If a [w,f,e] sequence is supplied, make it into a Spectrum object
        elif isinstance(spectrum, (list, tuple, np.ndarray)):
            data = {'spectrum': Spectrum(spectrum, header=header), 'wavelength_units': '', 'flux_units': ''}
            try:
                data['header'] = list(map(list, data['spectrum'].header.cards))
            except:
                data['header'] = ''

        if data:
            fn = filepath if filepath.endswith('.txt') else filepath + 'spectrum.txt'

            # Write the header
            if data['header']:
                for n, line in enumerate(data['header']):
                    data['header'][n] = ['# {}'.format(str(line[0])).ljust(10)[:10],
                                         '{:50s} / {}'.format(*map(str, line[1:]))]
                try:
                    ii.write([np.asarray(i) for i in np.asarray(data['header']).T], fn, delimiter='\t',
                             format='no_header')
                except IOError:
                    pass

            # Write the data
            names = ['# wavelength [{}]'.format(data['wavelength_units']), 'flux [{}]'.format(data['flux_units'])]
            if len(data['spectrum'].data) == 3:
                if type(data['spectrum'].data[2]) in [np.ndarray, list]:
                    names += ['unc [{}]'.format(data['flux_units'])]
                else:
                    data['spectrum'].data = data['spectrum'].data[:2]

            with open(fn, mode='a') as f:
                ii.write([np.asarray(i, dtype=np.float64) for i in data['spectrum'].data], f, names=names,
                         delimiter='\t')

        else:
            print("Could not output spectrum: {}".format(spectrum))