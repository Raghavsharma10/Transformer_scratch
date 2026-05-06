def new_tmp(self):
        """ Create a new temp file allocation """

        self.tmp_idx += 1
        return p.join(self.tmp_dir, 'tmp_' + str(self.tmp_idx))