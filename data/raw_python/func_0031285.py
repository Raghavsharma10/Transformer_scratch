def get_term_by_name(self, name):
        """Get the GO term with the given GO term name.

        If the given name is not associated with any GO term, the function will
        search for it among synonyms.

        Parameters
        ----------
        name: str
            The name of the GO term.

        Returns
        -------
        `GOTerm`
            The GO term with the given name.

        Raises
        ------
        ValueError
            If the given name is found neither among the GO term names, nor
            among synonyms.
        """
        term = None
        try:
            term = self.terms[self.name2id[name]]
        except KeyError:
            try:
                term = self.terms[self.syn2id[name]]
            except KeyError:
                pass
            else:
                logger.info('GO term name "%s" is a synonym for "%s".',
                            name, term.name)

        if term is None:
            raise ValueError('GO term name "%s" not found!' % name)

        return term