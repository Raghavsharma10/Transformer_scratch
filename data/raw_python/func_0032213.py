def _checkContactType(self, contactType):
        """
        Possibly emit some warnings about C{contactType}'s implementation of
        L{IContactType}.

        @type contactType: L{IContactType} provider
        """
        if getattr(contactType, 'getEditFormForPerson', None) is None:
            warn(
                "IContactType now has the 'getEditFormForPerson'"
                " method, but %s did not implement it." % (
                    contactType.__class__,),
                category=PendingDeprecationWarning)

        if getattr(contactType, 'getEditorialForm', None) is not None:
            warn(
                "The IContactType %s defines the 'getEditorialForm'"
                " method, which is deprecated.  'getEditFormForPerson'"
                " does something vaguely similar." % (contactType.__class__,),
                category=DeprecationWarning)