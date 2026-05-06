def _store(self):
        """Store currently loaded filelist for instrument onto filesystem"""

        name = self.stored_file_name
        # check if current file data is different than stored file list
        # if so, move file list to previous file list, store current to file
        # if not, do nothing
        stored_files = self._load()
        if len(stored_files) != len(self.files):
            # # of items is different, things are new
            new_flag = True
        elif len(stored_files) == len(self.files):
            # # of items equal, check specifically for equality
            if stored_files.eq(self.files).all():
                new_flag = False
            else:
                # not equal, there are new files
                new_flag = True

        if new_flag:
            
            if self.write_to_disk:
                stored_files.to_csv(os.path.join(self.home_path,
                                                 'previous_'+name),
                                    date_format='%Y-%m-%d %H:%M:%S.%f')
                self.files.to_csv(os.path.join(self.home_path, name),
                                date_format='%Y-%m-%d %H:%M:%S.%f')
            else:
                self._previous_file_list = stored_files
                self._current_file_list = self.files.copy()
        return