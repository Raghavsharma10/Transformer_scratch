def tcmap(self):
        ''' Create a tokens-concepts map '''
        tcmap = dd(list)
        for concept in self.__concept_map.values():
            for w in concept.tokens:
                tcmap[w].append(concept)
        return tcmap