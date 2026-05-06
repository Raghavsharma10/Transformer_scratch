def addReadGroup(self, readGroup):
        """
        Adds the specified ReadGroup to this ReadGroupSet.
        """
        id_ = readGroup.getId()
        self._readGroupIdMap[id_] = readGroup
        self._readGroupIds.append(id_)