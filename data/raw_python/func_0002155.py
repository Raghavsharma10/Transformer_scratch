def _clean_header_df(self, df):
        """Format the header dataframe and add units."""
        if self.suffix == '-drvd.txt':
            df.units = {'release_time': 'second',
                        'precipitable_water': 'millimeter',
                        'inv_pressure': 'hPa',
                        'inv_height': 'meter',
                        'inv_strength': 'Kelvin',
                        'mixed_layer_pressure': 'hPa',
                        'mixed_layer_height': 'meter',
                        'freezing_point_pressure': 'hPa',
                        'freezing_point_height': 'meter',
                        'lcl_pressure': 'hPa',
                        'lcl_height': 'meter',
                        'lfc_pressure': 'hPa',
                        'lfc_height': 'meter',
                        'lnb_pressure': 'hPa',
                        'lnb_height': 'meter',
                        'lifted_index': 'degC',
                        'showalter_index': 'degC',
                        'k_index': 'degC',
                        'total_totals_index': 'degC',
                        'cape': 'Joule / kilogram',
                        'convective_inhibition': 'Joule / kilogram'}

        else:
            df.units = {'release_time': 'second',
                        'latitude': 'degrees',
                        'longitude': 'degrees'}

        return df