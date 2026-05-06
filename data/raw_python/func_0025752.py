def listTheExtras(self, deleteAlso):
        """ Use ConfigObj's get_extra_values() call to find any extra/unknown
        parameters we may have loaded.  Return a string similar to findTheLost.
        If deleteAlso is True, this will also delete any extra/unknown items.
        """
        # get list of extras
        extras = configobj.get_extra_values(self)
        # extras is in format: [(sections, key), (sections, key), ]
        # but we need: [(sections, key, result), ...] - set all results to
        # a bool just to make it the right shape.  BUT, since we are in
        # here anyway, make that bool mean something - hide info in it about
        # whether that extra item is a section (1) or just a single par (0)
        #
        # simplified, this is:  expanded = [ (x+(abool,)) for x in extras]
        expanded = [ (x+ \
                       ( bool(len(x[0])<1 and hasattr(self[x[1]], 'keys')), ) \
                     ) for x in extras]
        retval = ''
        if expanded:
            retval = flattened2str(expanded, extra=1)
        # but before we return, delete them (from ourself!) if requested to
        if deleteAlso:
            for tup_to_del in extras:
                target = self
                # descend the tree to the dict where this items is located.
                # (this works because target is not a copy (because the dict
                #  type is mutable))
                location = tup_to_del[0]
                for subdict in location: target = target[subdict]
                # delete it
                target.pop(tup_to_del[1])

        return retval