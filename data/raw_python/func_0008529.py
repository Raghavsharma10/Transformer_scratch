def _do_relation(self):
        """ Attaches subjects, objects and verbs.
            If the previous chunk is a subject/object/verb, it is stored in Sentence.relations{}.
        """
        if self.chunks:
            ch = self.chunks[-1]
            for relation, role in ch.relations:
                if role == "SBJ" or role == "OBJ":
                    self.relations[role][relation] = ch
            if ch.type in ("VP",):
                self.relations[ch.type][ch.relation] = ch