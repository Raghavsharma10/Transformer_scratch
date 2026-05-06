def _build_flags(self):
        """
        Function builds kwargs variable for run_window
        """
        # Check if all entries for selected arguments are nonempty
        for arg_dict in [x for x in self.args.values() if self.arg_is_selected(x)]:
            if 'entry' in arg_dict and not arg_dict['entry'].get_text():
                self.gui_helper.execute_dialog("Entry {0} is empty".format(arg_dict['label']))
                return False

        # Check for active CheckButtons
        for arg_dict in [x for x in self.args.values() if self.arg_is_selected(x)]:
            arg_name = arg_dict['arg'].get_dest()
            if 'entry' in arg_dict:
                self.kwargs[arg_name] = arg_dict['entry'].get_text()
            else:
                if arg_dict['arg'].get_gui_hint('type') == 'const':
                    self.kwargs[arg_name] = arg_dict['arg'].kwargs['const']
                else:
                    self.kwargs[arg_name] = True

        # Check for non active CheckButtons but with defaults flag
        for arg_dict in [x for x in self.args.values() if not self.arg_is_selected(x)]:
            arg_name = arg_dict['arg'].get_dest()
            if 'default' in arg_dict['arg'].kwargs:
                self.kwargs[arg_name] = arg_dict['arg'].get_gui_hint('default')
            elif arg_name in self.kwargs:
                del self.kwargs[arg_name]

        return True