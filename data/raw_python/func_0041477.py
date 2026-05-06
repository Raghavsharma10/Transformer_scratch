def create_templates(self, templates):
        """
        Gets a list of templates to insert into the database
        """
        count = 0
        for template in templates:
            if not self.template_exists_db(template):
                name, location, description, language = template
                text = self.open_file(location)
                html_content = self.get_html_content(text)
                data = {
                    'name': utils.camel_to_snake(name).upper(),
                    'html_content': html_content,
                    'content': self.text_version(html_content),
                    'subject': self.get_subject(text),
                    'description': description,
                    'language': language
                }
                if models.EmailTemplate.objects.create(**data):
                    count += 1
        return count