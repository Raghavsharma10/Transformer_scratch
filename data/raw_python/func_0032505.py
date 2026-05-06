def compact(self):
        """
        Switch to the compact variant of the live form template.

        By default, this will simply create a loader for the
        C{self.compactFragmentName} template and compact all of this form's
        parameters.
        """
        self.docFactory = webtheme.getLoader(self.compactFragmentName)
        for param in self.parameters:
            param.compact()