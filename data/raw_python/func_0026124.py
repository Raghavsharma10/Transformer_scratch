def popupChoices(self, event=None):
        """Popup right-click menu of special parameter operations

        Relies on browserEnabled, clearEnabled, unlearnEnabled, helpEnabled
        instance attributes to determine which items are available.
        """
        # don't bother if all items are disabled
        if NORMAL not in (self.browserEnabled, self.clearEnabled,
                          self.unlearnEnabled, self.helpEnabled):
            return

        self.menu = Menu(self.entry, tearoff = 0)
        if self.browserEnabled != DISABLED:
            # Handle file and directory in different functions (tkFileDialog)
            if capable.OF_TKFD_IN_EPAR:
                self.menu.add_command(label   = "File Browser",
                                      state   = self.browserEnabled,
                                      command = self.fileBrowser)
                self.menu.add_command(label   = "Directory Browser",
                                      state   = self.browserEnabled,
                                      command = self.dirBrowser)
            # Handle file and directory in the same function (filedlg)
            else:
                self.menu.add_command(label   = "File/Directory Browser",
                                      state   = self.browserEnabled,
                                      command = self.fileBrowser)
            self.menu.add_separator()
        self.menu.add_command(label   = "Clear",
                              state   = self.clearEnabled,
                              command = self.clearEntry)
        self.menu.add_command(label   = self.defaultsVerb,
                              state   = self.unlearnEnabled,
                              command = self.unlearnValue)
        self.menu.add_command(label   = 'Help',
                              state   = self.helpEnabled,
                              command = self.helpOnParam)

        # Get the current y-coordinate of the Entry
        ycoord = self.entry.winfo_rooty()

        # Get the current x-coordinate of the cursor
        xcoord = self.entry.winfo_pointerx() - XSHIFT

        # Display the Menu as a popup as it is not associated with a Button
        self.menu.tk_popup(xcoord, ycoord)