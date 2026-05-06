def enum(self, desc, func=None, args=None, krgs=None):
        """Add a menu entry whose name will be an auto indexed number."""
        name = str(len(self.entries)+1)
        self.entries.append(MenuEntry(name, desc, func, args or [], krgs or {}))