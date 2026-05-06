def close(self):
        """ Close this bpch file.

        """

        if not self.fp.closed:
            for v in list(self.var_data):
                del self.var_data[v]

            self.fp.close()