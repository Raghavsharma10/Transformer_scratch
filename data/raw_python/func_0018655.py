def process_data(self, stream, metadata):
        """
        Extract the tabulated data from the input file

        # Arguments
        stream (Streamlike object): A Streamlike object (nominally StringIO)
            containing the table to be extracted
        metadata (dict): metadata read in from the header and the namelist

        # Returns
        df (pandas.DataFrame): contains the data, processed to the standard
            MAGICCData format
        metadata (dict): updated metadata based on the processing performed
        """
        index = np.arange(metadata["firstyear"], metadata["lastyear"] + 1)

        # The first variable is the global values
        globe = stream.read_chunk("d")

        assert len(globe) == len(index)

        regions = stream.read_chunk("d")
        num_regions = int(len(regions) / len(index))
        regions = regions.reshape((-1, num_regions), order="F")

        data = np.concatenate((globe[:, np.newaxis], regions), axis=1)

        df = pd.DataFrame(data, index=index)

        if isinstance(df.index, pd.core.indexes.numeric.Float64Index):
            df.index = df.index.to_series().round(3)

        df.index.name = "time"

        regions = [
            "World",
            "World|Northern Hemisphere|Ocean",
            "World|Northern Hemisphere|Land",
            "World|Southern Hemisphere|Ocean",
            "World|Southern Hemisphere|Land",
        ]

        variable = convert_magicc6_to_magicc7_variables(
            self._get_variable_from_filepath()
        )
        variable = convert_magicc7_to_openscm_variables(variable)
        column_headers = {
            "variable": [variable] * (num_regions + 1),
            "region": regions,
            "unit": ["unknown"] * len(regions),
            "todo": ["SET"] * len(regions),
        }

        return df, metadata, self._set_column_defaults(column_headers)