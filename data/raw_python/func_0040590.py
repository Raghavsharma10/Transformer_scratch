def write_to_file(path_file, callback):
        """
        Reads the content of the given file, puts the content into the callback and writes the result back to the file.
        :param path_file: the path to the file
        :type path_file: str
        :param callback: the callback function
        :type callback: function
        """
        try:
            with open(path_file, 'r+', encoding='utf-8') as f:
                content = callback(f.read())
                f.seek(0)
                f.write(content)
                f.truncate()
        except Exception as e:
            raise Exception(
                'unable to minify file %(file)s, exception was %(exception)r' % {
                    'file': path_file,
                    'exception': e,
                }
            )