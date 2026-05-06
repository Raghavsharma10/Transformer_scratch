def _read(self, filename):
        """Internal routine that reads all data from the punch file."""
        data = {}
        parsers = [
            FirstDataParser(), CoordinateParser(), EnergyGradParser(),
            SkipApproxHessian(), HessianParser(), MassParser(),
        ]
        with open(filename) as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                # at each line, a parsers checks if it has to process a piece of
                # file. If that happens, the parser gets control over the file
                # and reads as many lines as it needs to collect data for some
                # attributes.
                for parser in parsers:
                    if parser.test(line, data):
                        parser.read(line, f, data)
                        break
        self.__dict__.update(data)