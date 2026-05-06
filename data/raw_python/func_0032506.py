def asSubForm(self, name):
        """
        Make a form suitable for nesting within another form (a subform) out
        of this top-level liveform.

        @param name: the name of the subform within its parent.
        @type name: C{unicode}

        @return: a subform.
        @rtype: L{LiveForm}
        """
        self.subFormName = name
        self.jsClass = _SUBFORM_JS_CLASS
        return self