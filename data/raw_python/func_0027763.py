def findOrCreate(self, userItemClass, __ifnew=None, **attrs):
        """
        Usage::

            s.findOrCreate(userItemClass [, function] [, x=1, y=2, ...])

        Example::

            class YourItemType(Item):
                a = integer()
                b = text()
                c = integer()

            def f(x):
                print x, \"-- it's new!\"
            s.findOrCreate(YourItemType, f, a=1, b=u'2')

        Search for an item with columns in the database that match the passed
        set of keyword arguments, returning the first match if one is found,
        creating one with the given attributes if not.  Takes an optional
        positional argument function to call on the new item if it is new.
        """
        andargs = []
        for k, v in attrs.iteritems():
            col = getattr(userItemClass, k)
            andargs.append(col == v)

        if len(andargs) == 0:
            cond = []
        elif len(andargs) == 1:
            cond = [andargs[0]]
        else:
            cond = [attributes.AND(*andargs)]

        for result in self.query(userItemClass, *cond):
            return result
        newItem = userItemClass(store=self, **attrs)
        if __ifnew is not None:
            __ifnew(newItem)
        return newItem