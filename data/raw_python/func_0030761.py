def _load_view(self, template_engine_name, template_dir):
        """
        Load view by name and return an instance.
        """
        file_name = template_engine_name.lower()
        class_name = "{}View".format(template_engine_name.title())
        try:
            view_module = import_module("rails.views.{}".format(file_name))
        except ImportError:
            raise Exception("Template engine '{}' not found in 'rails.views'".format(file_name))
        view_class = getattr(view_module, class_name)
        return view_class(template_dir)