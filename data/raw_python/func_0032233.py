def getPluginWidget(self, pluginName):
        """
        Return the named plugin's view.

        @type pluginName: C{unicode}
        @param pluginName: The name of the plugin.

        @rtype: L{LiveElement}
        """
        # this will always pick the first plugin with pluginName if there is
        # more than one.  don't do that.
        for plugin in self.plugins:
            if _organizerPluginName(plugin) == pluginName:
                view = self._toLiveElement(
                    plugin.personalize(self.person))
                view.setFragmentParent(self)
                return view