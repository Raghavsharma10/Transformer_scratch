def equipped(self):
        """ Returns a dict of classes that have the item equipped and in what slot """
        equipped = self._item.get("equipped", [])

        # WORKAROUND: 0 is probably an off-by-one error
        # WORKAROUND: 65535 actually serves a purpose (according to Valve)
        return dict([(eq["class"], eq["slot"]) for eq in equipped if eq["class"] != 0 and eq["slot"] != 65535])