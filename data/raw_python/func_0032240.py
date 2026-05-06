def peopleTable(self, request, tag):
        """
        Return a L{PersonScrollingFragment} which will display the L{Person}
        items in the wrapped organizer.
        """
        f = PersonScrollingFragment(
            self.organizer, None, Person.name, self.wt)
        f.setFragmentParent(self)
        f.docFactory = webtheme.getLoader(f.fragmentName)
        return f