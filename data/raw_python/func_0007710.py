def change_enables(self, enables_list: list):
        """
        Enable/disable inputs.

        :param enables_list: list containing enables for each key
        :return: None
        """
        for i, entry in enumerate(self.values):
            if enables_list[i]:
                entry.config(state=tk.NORMAL)
            else:
                entry.config(state=tk.DISABLED)