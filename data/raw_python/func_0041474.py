def get_plugs_mail_classes(self, app):
        """
        Returns a list of tuples, but it should
        return a list of dicts
        """
        classes = []
        members = self.get_members(app)
        for member in members:
            name, cls = member
            if inspect.isclass(cls) and issubclass(cls, PlugsMail) and name != 'PlugsMail':
                files_ = self.get_template_files(app.__file__, name)
                for file_ in files_:
                    try:
                        description = cls.description
                        location = file_
                        language = self.get_template_language(location)
                        classes.append((name, location, description, language))
                    except AttributeError:
                        raise AttributeError('Email class must specify email description.')
        return classes