def _read_index_file(self):
        """read the param_file, index_dir should already be set """
        param_file = os.path.join(self.index_dir, self.param_file)
        with open(param_file) as f:
            for line in f.readlines():
                (name, fasta_file, index_file, line_size, total_size) = line.strip().split("\t")
                self.size[name] = int(total_size)
                self.fasta_file[name] = fasta_file
                self.index_file[name] = index_file
                self.line_size[name] = int(line_size)