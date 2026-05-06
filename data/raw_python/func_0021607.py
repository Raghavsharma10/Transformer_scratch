def getSectionIndexByName(self, name):
        """
        Given a string representing a section name, tries to find the section index.

        @type name: str
        @param name: A section name.

        @rtype: int
        @return: The index, starting at 0, of the section.
        """
        index = -1
        
        if name:
            for i in range(len(self.sectionHeaders)):
                if self.sectionHeaders[i].name.value.find(name) >= 0:
                    index = i
                    break
        return index