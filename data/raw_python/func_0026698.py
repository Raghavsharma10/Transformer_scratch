def pretty(self):
        """Return a 'pretty' string representation of this `Key`.

        note: do not override the builtin `__str__` or `__repr__` methods!
        """
        retval = ("Key(name={}, type={}, listable={}, compare={}, "
                  "priority={}, kind_preference={}, "
                  "replace_better={})").format(
                      self.name, self.type, self.listable, self.compare,
                      self.priority, self.kind_preference, self.replace_better)
        return retval