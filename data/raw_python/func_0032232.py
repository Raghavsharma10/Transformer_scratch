def pluginTabbedPane(self, request, tag):
        """
        Render a L{tabbedPane.TabbedPaneFragment} with an entry for each item
        in L{plugins}.
        """
        iq = inevow.IQ(tag)
        tabNames = [
            _organizerPluginName(p).encode('ascii') # gunk
                for p in self.plugins]
        child = tabbedPane.TabbedPaneFragment(
            zip(tabNames,
                ([self.getPluginWidget(tabNames[0])]
                    + [iq.onePattern('pane-body') for _ in tabNames[1:]])))
        child.jsClass = u'Mantissa.People.PluginTabbedPane'
        child.setFragmentParent(self)
        return child