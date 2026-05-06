def copy(self, new_path, replace=False):
        """ Uses shutil to copy a file over """
        new_full_path = os.path.join(self.static_path, new_path)
        if replace or not os.path.exists(new_full_path):
            shutil.copy2(self.full_path, new_full_path)
            return True
        return False