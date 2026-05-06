def read_file(self):
        """
        Open the file and assiging the permission to read/write and
        return the content in json formate.

        Return : json data
        """
        file_obj = open(self.file, 'r')
        content = file_obj.read()
        file_obj.close()
        if content:
            content = json.loads(content)
            return content
        else:
            return {}