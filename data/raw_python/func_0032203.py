def getParameters(self, contactItem):
        """
        Return a list containing a single parameter suitable for changing the
        VIP status of a person.

        @type contactItem: L{_PersonVIPStatus}

        @rtype: C{list} of L{liveform.Parameter}
        """
        isVIP = False # default
        if contactItem is not None:
            isVIP = contactItem.person.vip
        return [liveform.Parameter(
            'vip', liveform.CHECKBOX_INPUT, bool, 'VIP', default=isVIP)]