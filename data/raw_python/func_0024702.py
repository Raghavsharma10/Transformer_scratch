def _validate_flux_unit(new_unit, wav_only=False):
        """Make sure flux unit is valid."""
        new_unit = units.validate_unit(new_unit)
        acceptable_types = ['spectral flux density wav',
                            'photon flux density wav']
        acceptable_names = ['PHOTLAM', 'FLAM']

        if not wav_only:  # Include per Hz units
            acceptable_types += ['spectral flux density',
                                 'photon flux density']
            acceptable_names += ['PHOTNU', 'FNU', 'Jy']

        if new_unit.physical_type not in acceptable_types:
            raise exceptions.SynphotError(
                'Source spectrum cannot operate in {0}. Acceptable units: '
                '{1}'.format(new_unit, ','.join(acceptable_names)))

        return new_unit