def get_finished(self):
        """ Clean up terminated processes and returns the list of their ids """
        indices  = []
        for idf, v in self.q.items():
            if v.poll() != None:
                indices.append(idf)

        for i in indices:
            self.q.pop(i)
        return indices