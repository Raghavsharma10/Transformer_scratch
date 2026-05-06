def get_template_files(self, location, class_name):
        """
        Multilanguage support means that for each template
        we can have multiple templtate files, this methods
        returns all the template (html and txt) files
        that match the (class) template name
        """
        template_name = utils.camel_to_snake(class_name)
        dir_ = location[:-9] + 'templates/emails/'
        files_ = []
        for file_ in self.get_templates_files_in_dir(dir_):
            if file_.startswith(template_name) and file_.endswith(('.html', '.txt')):
                if file_ in self.overrides:
                    files_.append(self.overrides[file_] + file_)
                else:
                    files_.append(dir_ + file_)
        return files_