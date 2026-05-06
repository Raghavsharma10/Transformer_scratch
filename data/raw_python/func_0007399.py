def clean_zipfile(self):
        '''remove existing zipfile'''
        if os.path.isfile(self.zip_file):
            os.remove(self.zip_file)