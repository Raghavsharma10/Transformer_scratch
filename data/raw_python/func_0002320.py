def render(self, name, value, attrs=None, renderer=None):
        """
        Render the placeholder field.
        """
        other_instance_languages = None
        if value and value != "-DUMMY-":
            if get_parent_language_code(self.parent_object):
                # Parent is a multilingual object, provide information
                # for the copy dialog.
                other_instance_languages = get_parent_active_language_choices(
                    self.parent_object, exclude_current=True)

        context = {
            'cp_plugin_list': list(self.plugins),
            'placeholder_id': '',
            'placeholder_slot': self.slot,
            'other_instance_languages': other_instance_languages,
        }
        return mark_safe(render_to_string('admin/fluent_contents/placeholderfield/widget.html', context))