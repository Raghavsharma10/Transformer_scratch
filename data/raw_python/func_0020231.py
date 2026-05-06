def count(self, start, stop):
        '''Count the number of elements bewteen *start* and *stop*.'''
        s1 = self.pickler.dumps(start)
        s2 = self.pickler.dumps(stop)
        return self.backend_structure().count(s1, s2)