def copy(self, path_patterns, symbolic_link=False, root=None,
             conflicts='fail'):
        ''' Copy the contents of a file to a new location, with target
        filename defined by the current File's entities and the specified
        path_patterns. '''
        new_filename = build_path(self.entities, path_patterns)
        if not new_filename:
            return None

        if new_filename[-1] == os.sep:
            new_filename += self.filename

        if isabs(self.path) or root is None:
            path = self.path
        else:
            path = join(root, self.path)

        if not exists(path):
            raise ValueError("Target filename to copy/symlink (%s) doesn't "
                             "exist." % path)

        if symbolic_link:
            contents = None
            link_to = path
        else:
            with open(path, 'r') as f:
                contents = f.read()
            link_to = None

        write_contents_to_file(new_filename, contents=contents,
                               link_to=link_to, content_mode='text', root=root,
                               conflicts=conflicts)