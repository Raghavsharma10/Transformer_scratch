def next_window(self, widget, data=None):
        """
        Function opens the run Window who executes the
        assistant project creation
        """
        # check whether deps-only is selected
        deps_only = ('deps_only' in self.args and self.args['deps_only']['checkbox'].get_active())

        # preserve argument value if it is needed to be preserved
        for arg_dict in [x for x in self.args.values() if 'preserved' in x['arg'].kwargs]:
            preserve_key = arg_dict['arg'].kwargs['preserved']
            # preserve entry text (string value)
            if 'entry' in arg_dict:
                if self.arg_is_selected(arg_dict):
                    config_manager.set_config_value(preserve_key, arg_dict['entry'].get_text())
            # preserve if checkbox is ticked (boolean value)
            else:
                config_manager.set_config_value(preserve_key, self.arg_is_selected(arg_dict))

        # save configuration into file
        config_manager.save_configuration_file()
        # get project directory and name
        project_dir = self.dir_name.get_text()
        full_name = self.get_full_dir_name()

        # check whether project directory and name is properly set
        if not deps_only and self.current_main_assistant.name == 'crt':
            if project_dir == "":
                return self.gui_helper.execute_dialog("Specify directory for project")
            else:
                # check whether directory is existing
                if not os.path.isdir(project_dir):
                    response = self.gui_helper.create_question_dialog(
                        "Directory {0} does not exists".format(project_dir),
                        "Do you want to create them?"
                    )
                    if response == Gtk.ResponseType.NO:
                        # User do not want to create a directory
                        return
                    else:
                        # Create directory
                        try:
                            os.makedirs(project_dir)
                        except OSError as os_err:
                            return self.gui_helper.execute_dialog("{0}".format(os_err))
                elif os.path.isdir(full_name):
                    return self.check_for_directory(full_name)

        if not self._build_flags():
            return

        if not deps_only and self.current_main_assistant.name == 'crt':
            self.kwargs['name'] = full_name
        self.kwargs['__ui__'] = 'gui_gtk+'

        self.data['kwargs'] = self.kwargs
        self.data['top_assistant'] = self.top_assistant
        self.data['current_main_assistant'] = self.current_main_assistant
        self.parent.run_window.open_window(widget, self.data)
        self.path_window.hide()