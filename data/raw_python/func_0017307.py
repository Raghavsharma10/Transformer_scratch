def get_data_from_intree(self):
        """
        Load *data* from a file or string and return as a list of strings.
        The data contents could be one newick string; a multiline NEXUS format
        for one tree; multiple newick strings on multiple lines; or multiple
        newick strings in a multiline NEXUS format. In any case, we will read
        in the data as a list on lines. 
        """

        # load string: filename or data stream
        if isinstance(self.intree, (str, bytes)):
            
            # strip it
            self.intree = self.intree.strip()

            # is a URL: make a list by splitting a string
            if any([i in self.intree for i in ("http://", "https://")]):
                response = requests.get(self.intree)
                response.raise_for_status()
                self.data = response.text.strip().split("\n")

            # is a file: read by lines to a list
            elif os.path.exists(self.intree):
                with open(self.intree, 'rU') as indata:
                    self.data = indata.readlines()

            # is a string: make into a list by splitting
            else:
                self.data = self.intree.split("\n")

        # load iterable: iterable of newick strings
        elif isinstance(self.intree, (list, set, tuple)):
            self.data = list(self.intree)