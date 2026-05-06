def write(self, filepath, magicc_version):
        """
        Write an input file to disk.

        Parameters
        ----------
        filepath : str
            Filepath of the file to write.

        magicc_version : int
            The MAGICC version for which we want to write files. MAGICC7 and MAGICC6
            namelists are incompatible hence we need to know which one we're writing
            for.
        """
        writer = determine_tool(filepath, "writer")(magicc_version=magicc_version)
        writer.write(self, filepath)