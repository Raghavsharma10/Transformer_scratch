def toContactEditorialParameter(self, contactType, person):
        """
        Convert the given contact type into a L{liveform.LiveForm} parameter.

        @type contactType: L{IContactType} provider.

        @type person: L{Person}

        @rtype: L{liveform.Parameter} or similar.
        """
        contactItems = list(contactType.getContactItems(person))
        if contactType.allowMultipleContactItems:
            defaults = []
            modelObjects = []
            for contactItem in contactItems:
                defaultedParameters = contactType.getParameters(contactItem)
                if defaultedParameters is None:
                    continue
                defaults.append(self._parametersToDefaults(
                    defaultedParameters))
                modelObjects.append(contactItem)
            descriptiveIdentifier = _descriptiveIdentifier(contactType)
            return liveform.ListChangeParameter(
                contactType.uniqueIdentifier(),
                contactType.getParameters(None),
                defaults=defaults,
                modelObjects=modelObjects,
                modelObjectDescription=descriptiveIdentifier)
        (contactItem,) = contactItems
        return liveform.FormParameter(
            contactType.uniqueIdentifier(),
            liveform.LiveForm(
                lambda **k: k,
                contactType.getParameters(contactItem)))