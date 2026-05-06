def get_stdouts(self):
        """ Get the list of stdout of each process """
        souts = []
        for v in self.q.values():
            souts.append(v.stdout)
        return souts