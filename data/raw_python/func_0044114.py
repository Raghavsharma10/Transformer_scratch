def __getTemplate(template_file_name):
        """Get temaplte to save the ranking.

        :param template_file_name: path to the template.
        :type template_file_name: str.

        :return: template for the file.
        :rtype: pystache's template.
        """
        with open(template_file_name) as template_file:
            template_raw = template_file.read()

        template = parse(template_raw)
        return template