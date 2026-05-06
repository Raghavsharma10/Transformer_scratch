def do_get(self, from_path, to_path):
        """
        Copy file from Ndrive to local file and print out out the metadata.

        Examples:
          Ndrive> get file.txt ~/ndrive-file.txt
        """
        to_file = open(os.path.expanduser(to_path), "wb")

        self.n.downloadFile(self.current_path + "/" + from_path, to_path)