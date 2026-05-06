def init_db(self):
    '''init_db for the filesystem ensures that the base folder (named 
       according to the studyid) exists.
    '''
    self.session = None

    if not os.path.exists(self.data_base):
        mkdir_p(self.data_base)

    self.database = "%s/%s" %(self.data_base, self.study_id)
    if not os.path.exists(self.database):
        mkdir_p(self.database)