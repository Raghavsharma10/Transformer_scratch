def _load_controllers(self):
        """
        Load all controllers from folder 'controllers'.

        Ignore files with leading underscore (for example: controllers/_blogs.py)
        """
        for file_name in os.listdir(os.path.join(self._project_dir, 'controllers')):
            # ignore disabled controllers
            if not file_name.startswith('_'):
                module_name = file_name.split('.', 1)[0]
                module_path = "controllers.{}".format(module_name)
                module = import_module(module_path)
                # transform 'blog_articles' file name to 'BlogArticles' class
                controller_class_name = module_name.title().replace('_', '')
                controller_class = getattr(module, controller_class_name)
                controller = controller_class()
                for action_name in dir(controller):
                    action = getattr(controller, action_name)
                    if action_name.startswith('_') or not callable(action):
                        continue
                    url_path = "/".join([module_name, action_name])
                    self._controllers[url_path] = action
        return self._controllers