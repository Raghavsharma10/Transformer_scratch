def run(self):
        """Run merging.
		"""

        print("", file=sys.stderr)
        print("Going to merge/convert RNF-FASTQ files.", file=sys.stderr)
        print("", file=sys.stderr)
        print("   mode:          ", self.mode, file=sys.stderr)
        print("   input files:   ", ", ".join(self.input_files_fn), file=sys.stderr)
        print("   output files:  ", ", ".join(self.output_files_fn), file=sys.stderr)
        print("", file=sys.stderr)

        while len(self.i_files_weighted) > 0:
            file_id = self.rng.randint(0, len(self.i_files_weighted) - 1)
            for i in range(READS_IN_GROUP * self._reads_in_tuple):
                if self.i_files_weighted[file_id].closed:
                    del self.i_files_weighted[file_id]
                    break

                ln1 = self.i_files_weighted[file_id].readline()
                ln2 = self.i_files_weighted[file_id].readline()
                ln3 = self.i_files_weighted[file_id].readline()
                ln4 = self.i_files_weighted[file_id].readline()

                if ln1 == "" or ln2 == "" or ln3 == "" or ln4 == "":
                    self.i_files_weighted[file_id].close()
                    del self.i_files_weighted[file_id]
                    break
                assert ln1[0] == "@", ln1
                assert ln3[0] == "+", ln3
                self.output.save_read(ln1, ln2, ln3, ln4)
        self.output.close()