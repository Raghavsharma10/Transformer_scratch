def write_dir_tree(self, tree):
        """ Recur through dir tree data structure and write it as a set of objects """

        dirs  = tree['dirs']; files = tree['files']
        child_dirs = {name : self.write_dir_tree(contents) for name, contents in dirs.iteritems()}
        return self.write_index_object('tree', {'files' : files, 'dirs': child_dirs})