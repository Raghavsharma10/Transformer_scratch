def fs_put(self, rpath, data):
        """ Add a file to the FS """
        try:
            self.begin()

            # Add the file to the fs
            self.file_put_contents(rpath, data)

            # Add to the manifest
            manifest = self.read_local_manifest()
            manifest['files'][rpath] = self.get_single_file_info(rpath)
            self.write_local_manifest(manifest)

            self.commit()
        except:
            self.rollback(); raise