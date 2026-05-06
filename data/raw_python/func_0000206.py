def get_id_by_name(self, obj, name):
        """ Function get_id_by_name
        Get the id of an object

        @param obj: object name ('hosts', 'puppetclasses'...)
        @param id: the id of the object (name or id)
        @return RETURN: the targeted object
        """
        list = self.list(obj, filter='name = "{}"'.format(name),
                         only_id=True, limit=1)
        return list[name] if name in list.keys() else False