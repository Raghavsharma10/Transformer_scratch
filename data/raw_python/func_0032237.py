def getImportPeople(self):
        """
        Return an L{ImportPeopleWidget} which is a child of this fragment and
        which will add people to C{self.organizer}.
        """
        fragment = ImportPeopleWidget(self.organizer)
        fragment.setFragmentParent(self)
        return fragment