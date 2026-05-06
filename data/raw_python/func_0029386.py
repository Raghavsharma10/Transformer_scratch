def add(self, name, desc, func=None, args=None, krgs=None):
        """Add a menu entry."""
        self.entries.append(MenuEntry(name, desc, func, args or [], krgs or {}))