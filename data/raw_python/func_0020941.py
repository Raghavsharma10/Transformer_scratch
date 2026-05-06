def changed_path(self):
        "Find any changed path and update all changed modification times."
        result = None  # default
        for path in self.paths_to_modification_times:
            lastmod = self.paths_to_modification_times[path]
            mod = os.path.getmtime(path)
            if mod > lastmod:
                result = "Watch file has been modified: " + repr(path)
            self.paths_to_modification_times[path] = mod
        for folder in self.folder_paths:
            for filename in os.listdir(folder):
                subpath = os.path.join(folder, filename)
                if os.path.isfile(subpath) and subpath not in self.paths_to_modification_times:
                    result = "New file in watched folder: " + repr(subpath)
                    self.add(subpath)
        if self.check_python_modules:
            # refresh the modules
            self.add_all_modules()
        if self.check_javascript:
            self.watch_javascript()
        return result