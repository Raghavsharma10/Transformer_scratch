def printSkosTree(self, element = None, showids=False, labels=False, showtype=False):
        """
        Print nicely into stdout the SKOS tree of an ontology

        Note: indentation is made so that ids up to 3 digits fit in, plus a space.
        [123]1--
        [1]123--
        [12]12--
        """
        TYPE_MARGIN = 13 # length for skos:concept

        if not element:	 # first time
            for x in self.toplayerSkosConcepts:
                printGenericTree(x, 0, showids, labels, showtype, TYPE_MARGIN)

        else:
            printGenericTree(element, 0, showids, labels, showtype, TYPE_MARGIN)