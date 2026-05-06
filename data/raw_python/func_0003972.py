def add_relation(self, source, destination):
        """Add new a relation to the bejection"""
        if self.in_sources(source):
            if self.forward[source] != destination:
                raise ValueError("Source is already in use. Destination does "
                                 "not match.")
            else:
                raise ValueError("Source-Destination relation already exists.")
        elif self.in_destinations(destination):
            raise ValueError("Destination is already in use. Source does not "
                             "match.")
        else:
            self.forward[source] = destination
            self.reverse[destination] = source