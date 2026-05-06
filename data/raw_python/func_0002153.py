def _get_fwf_params(self):
        """Produce a dictionary with names, colspecs, and dtype for IGRA2 data.

        Returns a dict with entries 'body' and 'header'.
        """
        def _cdec(power=1):
            """Make a function to convert string 'value*10^power' to float."""
            def _cdec_power(val):
                if val in ['-9999', '-8888', '-99999']:
                    return np.nan
                else:
                    return float(val) / 10**power
            return _cdec_power

        def _cflag(val):
            """Replace alphabetic flags A and B with numeric."""
            if val == 'A':
                return 1
            elif val == 'B':
                return 2
            else:
                return 0

        def _ctime(strformat='MMMSS'):
            """Return a function converting a string from MMMSS or HHMM to seconds."""
            def _ctime_strformat(val):
                time = val.strip().zfill(5)

                if int(time) < 0:
                    return np.nan
                elif int(time) == 9999:
                    return np.nan
                else:
                    if strformat == 'MMMSS':
                        minutes = int(time[0:3])
                        seconds = int(time[3:5])
                        time_seconds = minutes * 60 + seconds
                    elif strformat == 'HHMM':
                        hours = int(time[0:2])
                        minutes = int(time[2:4])
                        time_seconds = hours * 3600 + minutes * 60
                    else:
                        sys.exit('Unrecognized time format')

                return time_seconds
            return _ctime_strformat

        def _clatlon(x):
            n = len(x)
            deg = x[0:n - 4]
            dec = x[n - 4:]
            return float(deg + '.' + dec)

        if self.suffix == '-drvd.txt':
            names_body = ['pressure', 'reported_height', 'calculated_height',
                          'temperature', 'temperature_gradient', 'potential_temperature',
                          'potential_temperature_gradient', 'virtual_temperature',
                          'virtual_potential_temperature', 'vapor_pressure',
                          'saturation_vapor_pressure', 'reported_relative_humidity',
                          'calculated_relative_humidity', 'u_wind', 'u_wind_gradient',
                          'v_wind', 'v_wind_gradient', 'refractive_index']

            colspecs_body = [(0, 7), (8, 15), (16, 23), (24, 31), (32, 39),
                             (40, 47), (48, 55), (56, 63), (64, 71), (72, 79),
                             (80, 87), (88, 95), (96, 103), (104, 111), (112, 119),
                             (120, 127), (128, 135), (137, 143), (144, 151)]

            conv_body = {'pressure': _cdec(power=2),
                         'reported_height': int,
                         'calculated_height': int,
                         'temperature': _cdec(),
                         'temperature_gradient': _cdec(),
                         'potential_temperature': _cdec(),
                         'potential_temperature_gradient': _cdec(),
                         'virtual_temperature': _cdec(),
                         'virtual_potential_temperature': _cdec(),
                         'vapor_pressure': _cdec(power=3),
                         'saturation_vapor_pressure': _cdec(power=3),
                         'reported_relative_humidity': _cdec(),
                         'calculated_relative_humidity': _cdec(),
                         'u_wind': _cdec(),
                         'u_wind_gradient': _cdec(),
                         'v_wind': _cdec(),
                         'v_wind_gradient': _cdec(),
                         'refractive_index': int}

            names_header = ['site_id', 'year', 'month', 'day', 'hour', 'release_time',
                            'number_levels', 'precipitable_water', 'inv_pressure',
                            'inv_height', 'inv_strength', 'mixed_layer_pressure',
                            'mixed_layer_height', 'freezing_point_pressure',
                            'freezing_point_height', 'lcl_pressure', 'lcl_height',
                            'lfc_pressure', 'lfc_height', 'lnb_pressure', 'lnb_height',
                            'lifted_index', 'showalter_index', 'k_index', 'total_totals_index',
                            'cape', 'convective_inhibition']

            colspecs_header = [(1, 12), (13, 17), (18, 20), (21, 23), (24, 26),
                               (27, 31), (31, 36), (37, 43), (43, 48), (49, 55),
                               (55, 61), (61, 67), (67, 73), (73, 79), (79, 85),
                               (85, 91), (91, 97), (97, 103), (103, 109), (109, 115),
                               (115, 121), (121, 127), (127, 133), (133, 139),
                               (139, 145), (145, 151), (151, 157)]

            conv_header = {'site_id': str,
                           'year': int,
                           'month': int,
                           'day': int,
                           'hour': int,
                           'release_time': _ctime(strformat='HHMM'),
                           'number_levels': int,
                           'precipitable_water': _cdec(power=2),
                           'inv_pressure': _cdec(power=2),
                           'inv_height': int,
                           'inv_strength': _cdec(),
                           'mixed_layer_pressure': _cdec(power=2),
                           'mixed_layer_height': int,
                           'freezing_point_pressure': _cdec(power=2),
                           'freezing_point_height': int,
                           'lcl_pressure': _cdec(power=2),
                           'lcl_height': int,
                           'lfc_pressure': _cdec(power=2),
                           'lfc_height': int,
                           'lnb_pressure': _cdec(power=2),
                           'lnb_height': int,
                           'lifted_index': int,
                           'showalter_index': int,
                           'k_index': int,
                           'total_totals_index': int,
                           'cape': int,
                           'convective_inhibition': int}

            na_vals = ['-99999']

        else:
            names_body = ['lvltyp1', 'lvltyp2', 'etime', 'pressure',
                          'pflag', 'height', 'zflag', 'temperature', 'tflag',
                          'relative_humidity', 'dewpoint_depression',
                          'direction', 'speed']

            colspecs_body = [(0, 1), (1, 2), (3, 8), (9, 15), (15, 16),
                             (16, 21), (21, 22), (22, 27), (27, 28),
                             (28, 33), (34, 39), (40, 45), (46, 51)]

            conv_body = {'lvltyp1': int,
                         'lvltyp2': int,
                         'etime': _ctime(strformat='MMMSS'),
                         'pressure': _cdec(power=2),
                         'pflag': _cflag,
                         'height': int,
                         'zflag': _cflag,
                         'temperature': _cdec(),
                         'tflag': _cflag,
                         'relative_humidity': _cdec(),
                         'dewpoint_depression': _cdec(),
                         'direction': int,
                         'speed': _cdec()}

            names_header = ['site_id', 'year', 'month', 'day', 'hour', 'release_time',
                            'number_levels', 'pressure_source_code',
                            'non_pressure_source_code',
                            'latitude', 'longitude']

            colspecs_header = [(1, 12), (13, 17), (18, 20), (21, 23), (24, 26),
                               (27, 31), (32, 36), (37, 45), (46, 54), (55, 62), (63, 71)]

            na_vals = ['-8888', '-9999']

            conv_header = {'release_time': _ctime(strformat='HHMM'),
                           'number_levels': int,
                           'latitude': _clatlon,
                           'longitude': _clatlon}

        return {'body': {'names': names_body,
                         'colspecs': colspecs_body,
                         'converters': conv_body,
                         'na_values': na_vals,
                         'index_col': False},
                'header': {'names': names_header,
                           'colspecs': colspecs_header,
                           'converters': conv_header,
                           'na_values': na_vals,
                           'index_col': False}}