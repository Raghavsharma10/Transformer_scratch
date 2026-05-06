def get_field_list(self):
        """
        Retrieve list of all fields currently configured
        """

        list_out = []
        for field in self.fields:
            list_out.append(field)

        return list_out