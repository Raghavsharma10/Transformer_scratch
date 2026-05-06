def read_file(filename):
        """
        Reads the lines of a file into a list, and returns the list
        :param filename: String - path and name of the file
        :return: List - lines within the file
        """
        lines = []
        with open(filename) as f:
            for line in f:
                if len(line.strip()) != 0:
                    lines.append(line.strip())
        return lines