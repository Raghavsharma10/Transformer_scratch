def generate_api_for_source(self, source_fpath: str):
        """Generate end json api file with directory structure for concrete
        source file."""
        content = self.convert_content(source_fpath)
        if content is None:
            return

        dest_fpath = self.dest_fpath(source_fpath)
        self.create_fpath_dir(dest_fpath)

        with open(dest_fpath, 'w+') as dest_f:
            json.dump(content, dest_f, cls=DateTimeJsonEncoder)