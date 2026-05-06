def from_file(filename):
        """Read in filename and creates a trace object.

        :param filename: path to nu(x|s)mv output file
        :type filename: str
        :return:
        """
        trace = Trace()
        reached = False
        with open(filename) as fp:
            for line in fp.readlines():
                if not reached and line.strip() == "Trace Type: Counterexample":
                    reached = True
                    continue
                elif reached:
                    trace.parse_line(line)
            return trace