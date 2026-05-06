def getParameters(self, notes):
        """
        Return a C{list} of one L{LiveForm} parameter for editing a
        L{Notes}.

        @type notes: L{Notes} or C{NoneType}
        @param notes: If not C{None}, an existing contact item from
            which to get the parameter's default value.

        @rtype: C{list}
        """
        defaultNotes = u''
        if notes is not None:
            defaultNotes = notes.notes
        return [
            liveform.Parameter('notes', liveform.TEXTAREA_INPUT,
                               unicode, 'Notes', default=defaultNotes)]