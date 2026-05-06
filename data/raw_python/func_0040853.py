def begin(self):
        """ Begin a transaction """

        if self.journal != None:
            raise Exception('Storage is already active, nested begin not supported')

        # under normal operation journal is deleted at end of transaction
        # if it does exist we need to roll back
        if os.path.isfile(self.j_file):  self.rollback()

        self.journal = open(self.j_file, 'w')