def actions(self):
        """Gets the list of allowed actions

        :rtype: list[str]
        """
        r = self.session.query(models.Action).all()
        return [x.type_name for x in r]