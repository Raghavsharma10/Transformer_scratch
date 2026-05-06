def git_root(self):
        """
        Find the root git folder
        """
        if not getattr(self, "_git_folder", None):
            root_folder = os.path.abspath(self.parent_dir)
            while not os.path.exists(os.path.join(root_folder, '.git')):
                if root_folder == '/':
                    raise HarpoonError("Couldn't find a .git folder", start_at=self.parent_dir)
                root_folder = os.path.dirname(root_folder)
            self._git_folder = root_folder
        return self._git_folder