def upgradeItem(self, thisItem):
        """
        Upgrade a legacy item.

        @raise axiom.errors.UpgraderRecursion: If the given item is already in
            the process of being upgraded.
        """
        sid = thisItem.storeID
        if sid in self._currentlyUpgrading:
            raise UpgraderRecursion()
        self._currentlyUpgrading[sid] = thisItem
        try:
            return upgradeAllTheWay(thisItem)
        finally:
            self._currentlyUpgrading.pop(sid)