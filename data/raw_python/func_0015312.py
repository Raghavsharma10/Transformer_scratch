def open_window(self, data=None):
        """
        Function opens the Options dialog
        """
        self.args = dict()
        if data is not None:
            self.top_assistant = data.get('top_assistant', None)
            self.current_main_assistant = data.get('current_main_assistant', None)
            self.kwargs = data.get('kwargs', None)
            self.data['debugging'] = data.get('debugging', False)
        project_dir = self.get_default_project_dir()
        self.dir_name.set_text(project_dir)
        self.label_full_prj_dir.set_text(project_dir)
        self.dir_name.set_sensitive(True)
        self.dir_name_browse_btn.set_sensitive(True)
        self._remove_widget_items()
        if self.current_main_assistant.name != 'crt' and self.project_name_shown:
            self.box6.remove(self.box_project)
            self.project_name_shown = False
        elif self.current_main_assistant.name == 'crt' and not self.project_name_shown:
            self.box6.remove(self.box_path_main)
            self.box6.pack_start(self.box_project, False, False, 0)
            self.box6.pack_end(self.box_path_main, False, False, 0)
            self.project_name_shown = True
        caption_text = "Project: "
        row = 0
        # get selectected assistants, but without TopAssistant itself
        path = self.top_assistant.get_selected_subassistant_path(**self.kwargs)[1:]
        caption_parts = []

        # Finds any dependencies
        found_deps = [x for x in path if x.dependencies()]
        # This bool variable is used for showing text "Available options:"
        any_options = False
        for assistant in path:
            caption_parts.append("<b>" + assistant.fullname + "</b>")
            for arg in sorted([x for x in assistant.args if not '--name' in x.flags], key=lambda y: y.flags):
                if not (arg.name == "deps_only" and not found_deps):
                    row = self._add_table_row(arg, len(arg.flags) - 1, row) + 1
                    any_options = True
        if not any_options:
            self.title.set_text("")
        else:
            self.title.set_text("Available options:")
        caption_text += ' -> '.join(caption_parts)
        self.label_caption.set_markup(caption_text)
        self.path_window.show_all()
        self.entry_project_name.set_text(os.path.basename(self.kwargs.get('name', '')))
        self.entry_project_name.set_sensitive(True)
        self.run_btn.set_sensitive(not self.project_name_shown or self.entry_project_name.get_text() != "")
        if 'name' in self.kwargs:
            self.dir_name.set_text(os.path.dirname(self.kwargs.get('name', '')))
        for arg_name, arg_dict in [(k, v) for (k, v) in self.args.items() if self.kwargs.get(k)]:
            if 'checkbox' in arg_dict:
                arg_dict['checkbox'].set_active(True)
            if 'entry' in arg_dict:
                arg_dict['entry'].set_sensitive(True)
                arg_dict['entry'].set_text(self.kwargs[arg_name])
            if 'browse_btn' in arg_dict:
                arg_dict['browse_btn'].set_sensitive(True)