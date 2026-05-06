def open_window(self, widget, data=None):
        """
        Function opens Main Window and in case of previously created
        project is switches to /home directory
        This is fix in case that da creats a project
        and project was deleted and GUI was not closed yet
        """
        if data is not None:
            self.data = data
        os.chdir(os.path.expanduser('~'))
        self.kwargs = dict()
        self.main_win.show_all()