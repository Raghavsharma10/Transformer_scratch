def _makeALiveForm(self, parameters, identifier, removable=True):
        """
        Make a live form with the parameters C{parameters}, which will be used
        to edit the values/model object with identifier C{identifier}.

        @type parameters: C{list}
        @param parameters: list of L{Parameter} instances.

        @type identifier: C{int}

        @type removable: C{bool}

        @rtype: L{repeatedLiveFormWrapper}
        """
        liveForm = self.liveFormFactory(lambda **k: None, parameters, self.name)
        liveForm = self._prepareSubForm(liveForm)
        liveForm = self.repeatedLiveFormWrapper(liveForm, identifier, removable)
        liveForm.docFactory = webtheme.getLoader(liveForm.fragmentName)
        return liveForm