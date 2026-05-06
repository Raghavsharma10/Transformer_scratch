def flatten_dir_tree(self, tree):
        """ Convert a file tree back into a flat dict """

        result = {}

        def helper(tree, leading_path = ''):
            dirs  = tree['dirs']; files = tree['files']
            for name, file_info in files.iteritems():
                file_info['path'] = leading_path + '/'  + name
                result[file_info['path']] = file_info

            for name, contents in dirs.iteritems():
                helper(contents, leading_path +'/'+ name)
        helper(tree); return result