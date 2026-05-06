def btn_clicked(self, widget, data=None):
        """
        Function is used for case that assistant does not have any
        subassistants
        """
        self.kwargs['subassistant_0'] = self.get_current_main_assistant().name
        self.kwargs['subassistant_1'] = data
        if 'subassistant_2' in self.kwargs:
            del self.kwargs['subassistant_2']
        self._open_path_window()