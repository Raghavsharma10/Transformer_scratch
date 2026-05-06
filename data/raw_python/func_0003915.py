def add_related(self, *objects):
        """Add related items

           The arguments can be individual items or cluster objects containing
           several items.

           When two groups of related items share one or more common members,
           they will be merged into one cluster.
        """
        master = None # this will become the common cluster of all related items
        slaves = set([]) # set of clusters that are going to be merged in the master
        solitaire = set([]) # set of new items that are not yet part of a cluster
        for new in objects:
            if isinstance(new, self.cls):
                if master is None:
                    master = new
                else:
                    slaves.add(new)
                for item in new.items:
                    existing = self.lookup.get(item)
                    if existing is not None:
                        slaves.add(existing)
            else:
                cluster = self.lookup.get(new)
                if cluster is None:
                    #print "solitaire", new
                    solitaire.add(new)
                elif master is None:
                    #print "starting master", new
                    master = cluster
                elif master != cluster:
                    #print "in slave", new
                    slaves.add(cluster)
                #else:
                    ##nothing to do
                    #print "new in master", new

        if master is None:
            master = self.cls([])

        for slave in slaves:
            master.update(slave)
        for item in solitaire:
            master.add_item(item)

        for item in master.items:
            self.lookup[item] = master