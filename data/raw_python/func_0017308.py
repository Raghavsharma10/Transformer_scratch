def parse_nexus(self):
        "get newick data from NEXUS"
        if self.data[0].strip().upper() == "#NEXUS":
            nex = NexusParser(self.data)
            self.data = nex.newicks
            self.tdict = nex.tdict