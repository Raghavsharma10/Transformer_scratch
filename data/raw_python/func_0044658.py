def write_content(self, content, destination):
        """
        Write given content to destination path.

        It will create needed directory structure first if it contain some
        directories that does not allready exists.

        Args:
            content (str): Content to write to target file.
            destination (str): Destination path for target file.

        Returns:
            str: Path where target file has been written.
        """
        directory = os.path.dirname(destination)

        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with io.open(destination, 'w', encoding='utf-8') as f:
            f.write(content)

        return destination