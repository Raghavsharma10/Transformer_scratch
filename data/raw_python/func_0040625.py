def read_local_manifest(self):
        """ Read the file manifest, or create a new one if there isn't one already """

        manifest = file_or_default(self.get_full_file_path(self.manifest_file), {
            'format_version' : 2,
            'root'           : '/',
            'have_revision'  : 'root',
            'files'          : {}}, json.loads)

        if 'format_version' not in manifest or manifest['format_version'] < 2:
            raise SystemExit('Please update the client manifest format')
        return manifest