def sub_menu_pressed(self, widget, event):
        """
        Function serves for getting full assistant path and
        collects the information from GUI
        """
        for index, data in enumerate(self.dev_assistant_path):
            index += 1
            if settings.SUBASSISTANT_N_STRING.format(index) in self.kwargs:
                del self.kwargs[settings.SUBASSISTANT_N_STRING.format(index)]
            self.kwargs[settings.SUBASSISTANT_N_STRING.format(index)] = data
        self.kwargs['subassistant_0'] = self.get_current_main_assistant().name
        self._open_path_window()