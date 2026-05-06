def store(self, transient_file, persistent_file):
        '''Makes PersistentFile from TransientFile'''
        #for i in range(5):
        #    persistent_file = PersistentFile(self.persistent_root,
        #                                     persistent_name, self)
        #    if not os.path.exists(persistent_file.path):
        #        break
        #else:
        #    raise Exception('Unable to find free file name')
        dirname = os.path.dirname(persistent_file.path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        os.rename(transient_file.path, persistent_file.path)
        return persistent_file