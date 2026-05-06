def open_file(self, file_):
        """
        Receives a file path has input and returns a
        string with the contents of the file
        """
        with open(file_, 'r', encoding='utf-8') as file:
            text = ''
            for line in file:
                text += line
        return text