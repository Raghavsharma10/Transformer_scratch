def write(self, magicc_input, filepath):
        """
        Write a MAGICC input file from df and metadata

        Parameters
        ----------
        magicc_input : :obj:`pymagicc.io.MAGICCData`
            MAGICCData object which holds the data to write

        filepath : str
            Filepath of the file to write to.
        """
        self._filepath = filepath
        # TODO: make copy attribute for MAGICCData
        self.minput = deepcopy(magicc_input)
        self.data_block = self._get_data_block()

        output = StringIO()

        output = self._write_header(output)
        output = self._write_namelist(output)
        output = self._write_datablock(output)

        with open(
            filepath, "w", encoding="utf-8", newline=self._newline_char
        ) as output_file:
            output.seek(0)
            copyfileobj(output, output_file)