def add_concept(self, concept_obj):
        ''' Add a concept to current concept list '''
        if concept_obj is None:
            raise Exception("Concept object cannot be None")
        elif concept_obj in self.__concepts:
            raise Exception("Concept object is already inside")
        elif concept_obj.cidx in self.__concept_map:
            raise Exception("Duplicated concept ID ({})".format(concept_obj.cidx))
        self.__concepts.append(concept_obj)
        self.__concept_map[concept_obj.cidx] = concept_obj
        concept_obj.sent = self
        return concept_obj