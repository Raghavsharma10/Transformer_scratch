def get_all_json_from_indexq(self):
        '''
        Gets all data from the todo files in indexq and returns one huge list of all data.
        '''
        files = self.get_all_as_list()
        out = []
        for efile in files:
            out.extend(self._open_file(efile))
        return out