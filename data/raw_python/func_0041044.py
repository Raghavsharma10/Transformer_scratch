def build_dir_tree(self, files):
        """ Convert a flat file dict into the tree format used for storage """

        def helper(split_files):
            this_dir = {'files' : {}, 'dirs' : {}}
            dirs = defaultdict(list)

            for fle in split_files:
                index = fle[0]; fileinfo = fle[1]
                if len(index)  == 1:
                    fileinfo['path'] = index[0] # store only the file name instead of the whole path
                    this_dir['files'][fileinfo['path']] = fileinfo
                elif len(index) > 1:
                    dirs[index[0]].append((index[1:], fileinfo))

            for name,info in dirs.iteritems():
                this_dir['dirs'][name] = helper(info)
            return this_dir
        return helper([(name.split('/')[1:], file_info) for name, file_info in files.iteritems()])