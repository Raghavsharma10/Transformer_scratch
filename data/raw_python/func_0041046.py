def read_dir_tree(self, file_hash):
        """ Recursively read the directory structure beginning at hash """

        json_d = self.read_index_object(file_hash, 'tree')
        node = {'files' : json_d['files'], 'dirs' : {}}
        for name, hsh in json_d['dirs'].iteritems(): node['dirs'][name] = self.read_dir_tree(hsh)
        return node