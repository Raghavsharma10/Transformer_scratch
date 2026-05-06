def make_po_file(self, potfile, locale):
        """
        Creates or updates the PO file for self.domain and :param locale:.
        Uses contents of the existing :param potfile:.

        Uses mguniq, msgmerge, and msgattrib GNU gettext utilities.
        """
        pofile = self._get_po_path(potfile, locale)

        msgs = self._get_unique_messages(potfile)
        msgs = self._merge_messages(potfile, pofile, msgs)
        msgs = self._strip_package_version(msgs)

        with open(pofile, 'w') as fp:
            fp.write(msgs)

        self._remove_obsolete_messages(pofile)