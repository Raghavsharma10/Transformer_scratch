def save_file(self, filename, text):
        """Save the given text under the given control filename and the
        current path."""
        if not filename.endswith('.py'):
            filename += '.py'
        path = os.path.join(self.currentpath, filename)
        with open(path, 'w', encoding="utf-8") as file_:
            file_.write(text)