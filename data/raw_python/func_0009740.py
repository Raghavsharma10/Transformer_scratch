def output(self, out_file):
        """Write the converted entries to out_file"""
        self.out_file = out_file
        out_file.write('event: ns : Nanoseconds\n')
        out_file.write('events: ns\n')
        self._output_summary()
        for entry in sorted(self.entries, key=_entry_sort_key):
            self._output_entry(entry)