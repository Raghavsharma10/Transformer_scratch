def file_name(self, event):
        """
        Helper method for determining the basename of the affected file.
        """
        name = os.path.basename(event.src_path)
        name = name.replace(".yaml", "")
        name = name.replace(".yml", "")

        return name