def getAddPerson(self):
        """
        Return an L{AddPersonFragment} which is a child of this fragment and
        which will add a person to C{self.organizer}.
        """
        fragment = AddPersonFragment(self.organizer)
        fragment.setFragmentParent(self)
        return fragment