def do_put(self, from_path, to_path):
        """
        Copy local file to Ndrive

        Examples:
          Ndrive> put ~/test.txt ndrive-copy-test.txt
        """
        from_file = open(os.path.expanduser(from_path), "rb")

        self.n.put(self.current_path + "/" + from_path, to_path)