def focus(self, focus: Optional[URIPARM]) -> None:
        """ Set the focus node(s).  If no focus node is specified, the evaluation will occur for all non-BNode
        graph subjects.  Otherwise it can be a string, a URIRef or a list of string/URIRef combinations

        :param focus: None if focus should be all URIRefs in the graph otherwise a URI or list of URI's
        """
        self._focus = normalize_uriparm(focus) if focus else None