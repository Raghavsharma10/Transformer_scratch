def coreferences(self):
        """
        Returns a list of Coreference classes

        :getter: Returns a list of coreferences
        :type: list of corenlp_xml.coreference.Coreference

        """
        if self._coreferences is None:
            coreferences = self._xml.xpath('/root/document/coreference/coreference')
            if len(coreferences) > 0:
                self._coreferences = [Coreference(self, element) for element in coreferences]
        return self._coreferences