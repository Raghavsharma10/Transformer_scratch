def _fuzz_data_file(self, data_file):
        """Generate fuzzed variant of given file.

        :param data_file: path to file to fuzz.
        :type data_file: str
        :return: path to fuzzed file.
        :rtype: str
        """
        buf = bytearray(open(os.path.abspath(data_file), 'rb').read())
        fuzzed = fuzzer(buf, self.fuzz_factor)
        try:
            _, fuzz_output = mkstemp(prefix='fuzzed_')
            open(fuzz_output, 'wb').write(fuzzed)
        finally:
            pass
        return fuzz_output