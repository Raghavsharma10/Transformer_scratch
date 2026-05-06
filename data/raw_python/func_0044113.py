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
        exportedUsers = self.getSortedUsers()
        template = self.__getTemplate(template_file_name)
        position = 1

        if not limit:
            exportedData["users"] = exportedUsers
        else:
            exportedData["users"] = exportedUsers[:limit]

        for u in exportedData["users"]:
            u["position"] = position
            u["comma"] = position < len(exportedData["users"])
            position += 1

        exportedData["extraData"] = data

        renderer = Renderer()
        output = renderer.render(template, exportedData)

        with open(output_file_name, "w") as text_file:
            text_file.write(output)