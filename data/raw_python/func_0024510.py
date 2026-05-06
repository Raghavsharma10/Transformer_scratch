def write_pkg_to_file(self, name, objects, path='.', filename=None):
        """Write a list of related objs to file"""
        # Kibana uses an array of docs, do the same
        # as opposed to a dict of docs
        pkg_objs = []
        for _, obj in iteritems(objects):
            pkg_objs.append(obj)
        sorted_pkg = sorted(pkg_objs, key=lambda k: k['_id'])
        output = self.json_dumps(sorted_pkg) + '\n'
        if filename is None:
            filename = self.safe_filename('Pkg', name)
        filename = os.path.join(path, filename)
        self.pr_inf("Writing to file: " + filename)
        with open(filename, 'w') as f:
            f.write(output)
        return filename