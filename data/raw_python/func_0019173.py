def prepare_network(self):
        """Load all network files as |Selections| (stored in module |pub|)
        and assign the "complete" selection to the |HydPy| object."""
        hydpy.pub.selections = selectiontools.Selections()
        hydpy.pub.selections += hydpy.pub.networkmanager.load_files()
        self.update_devices(hydpy.pub.selections.complete)