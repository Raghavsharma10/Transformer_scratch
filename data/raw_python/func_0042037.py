def _update_sys_path(self, package_path=None):
        """Updates and adds current directory to sys path"""
        self.package_path = package_path
        if not self.package_path in sys.path:
            sys.path.append(self.package_path)