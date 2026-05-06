def forms(self, req, tag):
        """
        Make and return some forms, using L{self.parameter.getInitialLiveForms}.

        @return: some subforms.
        @rtype: C{list} of L{LiveForm}
        """
        liveForms = self.parameter.getInitialLiveForms()
        for liveForm in liveForms:
            liveForm.setFragmentParent(self)
        return liveForms