def nextConcept(self, concepturi):
        """Returns the next skos concept in the list of concepts. If it's the last one, returns the first one."""
        if concepturi == self.skosConcepts[-1].uri:
            return self.skosConcepts[0]
        flag = False
        for x in self.skosConcepts:
            if flag == True:
                return x
            if x.uri == concepturi:
                flag = True
        return None