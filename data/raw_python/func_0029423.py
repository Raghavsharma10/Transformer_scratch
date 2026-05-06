def get_ancestors(self):
        "Returns the list of ancestor resources."

        if self._ancestors:
            return self._ancestors

        if not self.parent:
            return []

        obj = self.resource_map.get(self.parent.uid)

        while obj and obj.member_name:
            self._ancestors.append(obj)
            obj = obj.parent

        self._ancestors.reverse()
        return self._ancestors