def _add_table_row(self, arg, number, row):
        """
        Function adds options to a grid
        """
        self.args[arg.name] = dict()
        self.args[arg.name]['arg'] = arg
        check_box_title = arg.flags[number][2:].title()
        self.args[arg.name]['label'] = check_box_title
        align = self.gui_helper.create_alignment()
        if arg.kwargs.get('required'):
            # If argument is required then red star instead of checkbox
            star_label = self.gui_helper.create_label('<span color="#FF0000">*</span>')
            star_label.set_padding(0, 3)
            label = self.gui_helper.create_label(check_box_title)
            box = self.gui_helper.create_box()
            box.pack_start(star_label, False, False, 6)
            box.pack_start(label, False, False, 6)
            align.add(box)
        else:
            chbox = self.gui_helper.create_checkbox(check_box_title)
            chbox.set_alignment(0, 0)
            if arg.name == "deps_only":
                chbox.connect("clicked", self._deps_only_toggled)
            else:
                chbox.connect("clicked", self._check_box_toggled, arg.name)
            align.add(chbox)
            self.args[arg.name]['checkbox'] = chbox
        if row == 0:
            self.grid.add(align)
        else:
            self.grid.attach(align, 0, row, 1, 1)
        label = self.gui_helper.create_label(arg.kwargs['help'], justify=Gtk.Justification.LEFT)
        label.set_alignment(0, 0)
        label.set_padding(0, 3)
        self.grid.attach(label, 1, row, 1, 1)
        label_check_box = self.gui_helper.create_label(name="")
        self.grid.attach(label_check_box, 0, row, 1, 1)
        if arg.get_gui_hint('type') not in ['bool', 'const']:
            new_box = self.gui_helper.create_box(spacing=6)
            entry = self.gui_helper.create_entry(text="")
            align = self.gui_helper.create_alignment()
            align.add(entry)
            new_box.pack_start(align, False, False, 6)
            align_btn = self.gui_helper.create_alignment()
            ''' If a button is needed please add there and in function
                _check_box_toggled
                Also do not forget to create a function for that button
                This can not be done by any automatic tool from those reasons
                Some fields needs a input user like user name for GitHub
                and some fields needs to have interaction from user like selecting directory
            '''
            entry.set_text(arg.get_gui_hint('default'))
            entry.set_sensitive(arg.kwargs.get('required') == True)

            if arg.get_gui_hint('type') == 'path':
                browse_btn = self.gui_helper.button_with_label("Browse")
                browse_btn.connect("clicked", self.browse_clicked, entry)
                browse_btn.set_sensitive(arg.kwargs.get('required') == True)
                align_btn.add(browse_btn)
                self.args[arg.name]['browse_btn'] = browse_btn
            elif arg.get_gui_hint('type') == 'str':
                if arg.name == 'github' or arg.name == 'github-login':
                    link_button = self.gui_helper.create_link_button(text="For registration visit GitHub Homepage",
                                                                     uri="https://www.github.com")
                    align_btn.add(link_button)
            new_box.pack_start(align_btn, False, False, 6)
            row += 1
            self.args[arg.name]['entry'] = entry
            self.grid.attach(new_box, 1, row, 1, 1)
        else:
            if 'preserved' in arg.kwargs and config_manager.get_config_value(arg.kwargs['preserved']):
                if 'checkbox' in self.args[arg.name]:
                    self.args[arg.name]['checkbox'].set_active(True)
        return row