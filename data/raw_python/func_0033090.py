def _format_output(self, outfile_name, out_type):
        """ Prepend proper output prefix to output filename """

        outfile_name = self._absolute(outfile_name)
        outparts = outfile_name.split("/")
        outparts[-1] = self._out_format % (out_type, outparts[-1] )

        return '/'.join(outparts)