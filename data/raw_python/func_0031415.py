def addTitle(self, title, titleAlignments):
        """
        Add a new title to self.

        @param title: A C{str} title.
        @param titleAlignments: An instance of L{TitleAlignments}.
        @raises KeyError: If the title is already present.
        """
        if title in self:
            raise KeyError('Title %r already present in TitlesAlignments '
                           'instance.' % title)
        else:
            self[title] = titleAlignments