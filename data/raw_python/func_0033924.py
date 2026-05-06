def edit(
        self, index, name=None, priority=None,
        comment=None, done=None, parent=None
    ):
        """Modifies :index: to specified data.

        Every argument, which is not None, will get changed.

        If parent is not None, the item will get reparented.
        Use parent=-1 or parent='' for reparenting to top-level.

        :index: Index of the item to edit.
        :name: New name.
        :priority: New priority.
        :comment: New comment.
        :done: Done mark.
        :parent: New parent.

        """
        if parent == -1:
            parent = ''
        parent = self._split(parent)
        index = self._split(index)
        item = self.data
        for j, c in enumerate(index):
            item = item[int(c) - 1]
            if j + 1 != len(index):
                item = item[4]
        if name is not None:
            item[0] = name
        if priority is not None:
            item[1] = priority
        if comment is not None:
            item[2] = comment
        if done is not None:
            item[3] = done
        if parent is not None and parent != index[:-1]:
            parentitem = self.data
            for c in parent:
                parentitem = parentitem[int(c) - 1][4]
            parentitem.append(item)
            parent = index[:-1]
            parentitem = self.data
            for c in parent:
                parentitem = parentitem[int(c) - 1][4]
            parentitem.remove(item)