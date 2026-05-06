def getContactCreationParameters(self):
        """
        Yield a L{Parameter} for each L{IContactType} known.

        Each yielded object can be used with a L{LiveForm} to create a new
        instance of a particular L{IContactType}.
        """
        for contactType in self.getContactTypes():
            if contactType.allowMultipleContactItems:
                descriptiveIdentifier = _descriptiveIdentifier(contactType)
                yield liveform.ListChangeParameter(
                    contactType.uniqueIdentifier(),
                    contactType.getParameters(None),
                    defaults=[],
                    modelObjects=[],
                    modelObjectDescription=descriptiveIdentifier)
            else:
                yield liveform.FormParameter(
                    contactType.uniqueIdentifier(),
                    liveform.LiveForm(
                        lambda **k: k,
                        contactType.getParameters(None)))