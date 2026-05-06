def tabbedPane(self, req, tag):
        """
        Render a tabbed pane tab for each top-level
        L{xmantissa.ixmantissa.IPreferenceCollection} tab
        """
        navigation = webnav.getTabs(self.aggregator.getPreferenceCollections())
        pages = list()
        for tab in navigation:
            f = inevow.IRenderer(
                    self.aggregator.store.getItemByID(tab.storeID))
            f.tab = tab
            if hasattr(f, 'setFragmentParent'):
                f.setFragmentParent(self)
            pages.append((tab.name, f))

        f = tabbedPane.TabbedPaneFragment(pages, name='preference-editor')
        f.setFragmentParent(self)
        return f