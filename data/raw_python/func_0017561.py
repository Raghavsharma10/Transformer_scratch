def update_file(self, content):
        """
        It will convert json content to json string and update into file.

        Return:
        Boolean True/False
        """
        updated_content = json.dumps(content)
        file_obj = open(self.file, 'r+')
        file_obj.write(str(updated_content))
        file_obj.close()
        return True