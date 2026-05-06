def add(self, name, priority=3, comment="", parent=""):
        """Adds new item to the model.

        Name argument may contain (ref:) syntax, which will be
        stripped down as needed.

        :parent: should have a form "<itemref>.<subitemref...>" (e.g. "1.1").

        :name: Name (with refs).
        :priority: Item's priority.
        :comment: Comment.
        :parent: Item's parent ("" for top-level item).

        """
        item = [name, priority, comment, False, []]
        data = self.data
        for c in self._split(parent):
            data = data[int(c) - 1][4]
        data.append(item)