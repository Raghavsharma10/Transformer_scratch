def export(self, template_file_name, output_file_name,
               sort="public", data=None, limit=0):
        """Export ranking to a file.

        Args:
            template_file_name (str): where is the template
                (moustache template)
            output_file_name (str): where create the file with the ranking
            sort (str): field to sort the users
        """
        exportedData = {}
        exportedUsers = self.__exportUsers(sort, limit)

        exportedData["users"] = exportedUsers
        exportedData["extraData"] = data

        with open(template_file_name) as template_file:
            template_raw = template_file.read()

        template = parse(template_raw)
        renderer = Renderer()

        output = renderer.render(template, exportedData)

        with open(output_file_name, "w") as text_file:
            text_file.write(output)