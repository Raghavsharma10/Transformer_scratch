def fragments(self, req, tag):
        """
        Render our preference collection, any child preference
        collections we discover by looking at self.tab.children,
        and any fragments returned by its C{getSections} method.

        Subtabs and C{getSections} fragments are rendered as fieldsets
        inside the parent preference collection's tab.
        """
        f = self._collectionToLiveform()
        if f is not None:
            yield tags.fieldset[tags.legend[self.tab.name], f]

        for t in self.tab.children:
            f = inevow.IRenderer(
                    self.collection.store.getItemByID(t.storeID))
            f.tab = t
            if hasattr(f, 'setFragmentParent'):
                f.setFragmentParent(self)
            yield f

        for f in self.collection.getSections() or ():
            f = ixmantissa.INavigableFragment(f)
            f.setFragmentParent(self)
            f.docFactory = getLoader(f.fragmentName)
            yield tags.fieldset[tags.legend[f.title], f]