def rename(self, oldkey, newkey):
        """
        Change a keyname to another, without changing position in sequence.

        Implemented so that transformations can be made on keys,
        as well as on values. (used by encode and decode)

        Also renames comments.
        """
        if oldkey in self.scalars:
            the_list = self.scalars
        elif oldkey in self.sections:
            the_list = self.sections
        else:
            raise KeyError('Key "%s" not found.' % oldkey)
        pos = the_list.index(oldkey)
        #
        val = self[oldkey]
        dict.__delitem__(self, oldkey)
        dict.__setitem__(self, newkey, val)
        the_list.remove(oldkey)
        the_list.insert(pos, newkey)
        comm = self.comments[oldkey]
        inline_comment = self.inline_comments[oldkey]
        del self.comments[oldkey]
        del self.inline_comments[oldkey]
        self.comments[newkey] = comm
        self.inline_comments[newkey] = inline_comment