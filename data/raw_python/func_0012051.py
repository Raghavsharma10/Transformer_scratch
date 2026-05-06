def selection_dialog(self, courses):
        """
        opens a curses/picker based interface to select courses that should be downloaded.
        """
        selected = list(filter(lambda x: x.course.id in self._settings["selected_courses"], courses))
        selection = Picker(
            title="Select courses to download",
            options=courses,
            checked=selected).getSelected()
        if selection:
            self._settings["selected_courses"] = list(map(lambda x: x.course.id, selection))
            self.save()
            log.info("Updated course selection")