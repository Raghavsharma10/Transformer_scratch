def repeatForm(self):
        """
        Make and return a form, using L{self.parameter.asLiveForm}.

        @return: a subform.
        @rtype: L{LiveForm}
        """
        liveForm = self.parameter.asLiveForm()
        liveForm.setFragmentParent(self)
        return liveForm