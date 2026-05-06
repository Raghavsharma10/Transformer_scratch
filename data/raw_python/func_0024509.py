def write_object_to_file(self, obj, path='.', filename=None):
        """Convert obj (dict) to json string and write to file"""
        output = self.json_dumps(obj) + '\n'
        if filename is None:
            filename = self.safe_filename(obj['_type'], obj['_id'])
        filename = os.path.join(path, filename)
        self.pr_inf("Writing to file: " + filename)
        with open(filename, 'w') as f:
            f.write(output)
        # self.pr_dbg("Contents: " + output)
        return filename