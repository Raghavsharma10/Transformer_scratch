def write_to_file(self, filename, filetype=None):
        """Write the relaxation to a file.

        :param filename: The name of the file to write to. The type can be
                         autodetected from the extension: .dat-s for SDPA,
                         .task for mosek, .csv for human readable format, or
                         .txt for a symbolic export
        :type filename: str.
        :param filetype: Optional parameter to define the filetype. It can be
                         "sdpa" for SDPA , "mosek" for Mosek, "csv" for
                         human readable format, or "txt" for a symbolic export.
        :type filetype: str.
        """
        if filetype == "txt" and not filename.endswith(".txt"):
            raise Exception("TXT files must have .txt extension!")
        elif filetype is None and filename.endswith(".txt"):
            filetype = "txt"
        else:
            return super(SteeringHierarchy, self).write_to_file(filename,
                                                                filetype=filetype)
        tempfile_ = tempfile.NamedTemporaryFile()
        tmp_filename = tempfile_.name
        tempfile_.close()
        tmp_dats_filename = tmp_filename + ".dat-s"
        write_to_sdpa(self, tmp_dats_filename)
        f = open(tmp_dats_filename, 'r')
        f.readline();f.readline();f.readline()
        blocks = ((f.readline().strip().split(" = ")[0])[1:-1]).split(", ")
        block_offset, matrix_size = [0], 0
        for block in blocks:
            matrix_size += abs(int(block))
            block_offset.append(matrix_size)
        f.readline()
        matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
        for line in f:
            entry = line.strip().split("\t")
            var, block = int(entry[0]), int(entry[1])-1
            row, column = int(entry[2]) - 1, int(entry[3]) - 1
            value = float(entry[4])
            offset = block_offset[block]
            matrix[offset+row][offset+column] = int(value*var)
            matrix[offset+column][offset+row] = int(value*var)
        f.close()
        f = open(filename, 'w')
        for matrix_line in matrix:
            f.write(str(matrix_line).replace('[', '').replace(']', '') + '\n')
        f.close()
        os.remove(tmp_dats_filename)