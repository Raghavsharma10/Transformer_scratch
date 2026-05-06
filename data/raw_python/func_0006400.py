def addEntry(self):
        """Add the `Plot pyBAR data`. entry to `Dataset` menu.
        """
        export_icon = QtGui.QIcon()
        pixmap = QtGui.QPixmap(os.path.join(PLUGINSDIR,
                                            'csv/icons/document-export.png'))
        export_icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.plot_action = QtGui.QAction(
            translate('PlotpyBARdata',
                      "Plot data with pyBAR plugin",
                      "Plot data with pyBAR plugin"),
            self,
            shortcut=QtGui.QKeySequence.UnknownKey, triggered=self.plot,
            icon=export_icon,
            statusTip=translate('PlotpyBARdata',
                                "Plotting of selected data with pyBAR",
                                "Status bar text for the Dataset -> Plot pyBAR data... action"))

        # Add the action to the Dataset menu
        menu = self.vtgui.dataset_menu
        menu.addSeparator()
        menu.addAction(self.plot_action)

        # Add the action to the leaf context menu
        cmenu = self.vtgui.leaf_node_cm
        cmenu.addSeparator()
        cmenu.addAction(self.plot_action)