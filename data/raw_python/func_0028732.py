def write(self, data_in, data_out, *args, **kwargs):
        """
        :param data_in:
        :type data_in: hepconverter.parsers.ParsedData
        :param data_out: filelike object
        :type data_out: file
        :param args:
        :param kwargs:
        """
        self._get_tables(data_in)

        self.file_emulation = False
        outputs = []
        self._prepare_outputs(data_out, outputs)
        output = outputs[0]
        for i in xrange(len(self.tables)):
            table = self.tables[i]

            self._write_table(output, table)

        if data_out != output and hasattr(data_out, 'write'):
            output.Flush()
            output.ReOpen('read')
            file_size = output.GetSize()
            buff = bytearray(file_size)
            output.ReadBuffer(buff, file_size)
            data_out.write(buff)

        if self.file_emulation:
            filename = output.GetName()
            output.Close()