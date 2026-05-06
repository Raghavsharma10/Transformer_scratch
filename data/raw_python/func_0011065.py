def write_to_file(self, filename, filetype=None):
        """Write the relaxation to a file.

        :param filename: The name of the file to write to. The type can be
                         autodetected from the extension: .dat-s for SDPA,
                         .task for mosek or .csv for human readable format.
        :type filename: str.
        :param filetype: Optional parameter to define the filetype. It can be
                         "sdpa" for SDPA , "mosek" for Mosek, or "csv" for
                         human readable format.
        :type filetype: str.
        """
        if filetype == "sdpa" and not filename.endswith(".dat-s"):
            raise Exception("SDPA files must have .dat-s extension!")
        if filetype == "mosek" and not filename.endswith(".task"):
            raise Exception("Mosek files must have .task extension!")
        elif filetype is None and filename.endswith(".dat-s"):
            filetype = "sdpa"
        elif filetype is None and filename.endswith(".csv"):
            filetype = "csv"
        elif filetype is None and filename.endswith(".task"):
            filetype = "mosek"
        elif filetype is None:
            raise Exception("Cannot detect filetype from extension!")

        if filetype == "sdpa":
            write_to_sdpa(self, filename)
        elif filetype == "mosek":
            task = convert_to_mosek(self)
            task.writedata(filename)
        elif filetype == "csv":
            write_to_human_readable(self, filename)
        else:
            raise Exception("Unknown filetype")