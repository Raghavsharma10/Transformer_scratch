def dock(window):
    """ Expecting a window to parent into a Nuke panel, that is dockable. """

    # Deleting existing dock
    # There is a bug where existing docks are kept in-memory when closed via UI
    if self._dock:
        print("Deleting existing dock...")
        parent = self._dock
        dialog = None
        stacked_widget = None
        main_windows = []

        # Getting dock parents
        while parent:
            if isinstance(parent, QtWidgets.QDialog):
                dialog = parent
            if isinstance(parent, QtWidgets.QStackedWidget):
                stacked_widget = parent
            if isinstance(parent, QtWidgets.QMainWindow):
                main_windows.append(parent)
            parent = parent.parent()

        dialog.deleteLater()

        if len(main_windows) > 1:
            # Then it's a floating window
            if stacked_widget.count() == 1:
                # Then it's empty and we can close it,
                # as is native Nuke UI behaviour
                main_windows[0].deleteLater()

    # Creating new dock
    pane = nuke.getPaneFor("Properties.1")
    widget_path = "pyblish_nuke.lib.pyblish_nuke_dockwidget"
    panel = nukescripts.panels.registerWidgetAsPanel(widget_path,
                                                     window.windowTitle(),
                                                     "pyblish_nuke.dock",
                                                     True).addToPane(pane)

    panel_widget = panel.customKnob.getObject().widget
    panel_widget.layout().addWidget(window)
    _nuke_set_zero_margins(panel_widget)
    self._dock = panel_widget

    return self._dock