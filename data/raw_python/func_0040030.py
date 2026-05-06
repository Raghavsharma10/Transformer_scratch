def tooltip_query(self, widget, x, y, keyboard_mode, tooltip):
        """
        Set tooltip which appears when you hover mouse curson onto icon in system panel.
        """
        tooltip.set_text(subprocess.getoutput("acpi"))
        return True