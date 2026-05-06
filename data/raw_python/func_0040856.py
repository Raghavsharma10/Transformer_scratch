def commit(self, cont = False):
        """ Finish a transaction """

        self.journal.close()
        self.journal = None
        os.remove(self.j_file)

        for itm in os.listdir(self.tmp_dir): os.remove(cpjoin(self.tmp_dir, itm))

        if cont is True: self.begin()