def compress(self, data_list):
        """
        Return the cleaned_data of the form, everything should already be valid
        """
        data = {}
        if data_list:
            return dict(
                (f.name, data_list[i]) for i, f in enumerate(self.form))
        return data