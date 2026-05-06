def clean_value(self):
        """
        Populates json serialization ready data.
        This is the method used to serialize and store the object data in to DB

        Returns:
            List of dicts.
        """
        result = []
        for mdl in self:
            result.append(super(ListNode, mdl).clean_value())
        return result