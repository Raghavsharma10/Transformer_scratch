def _merge_files(self, input_files, output_file):
        """Combine the input files to a big output file"""
        # we assume that all the input files have the same charset
        with open(output_file, mode='wb') as out:
            for input_file in input_files:
                out.write(open(input_file, mode='rb').read())