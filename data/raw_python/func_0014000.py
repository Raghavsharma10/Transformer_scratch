def keep(self, keep_names):
        """Keeps variables (keep_names) while dropping other parameters"""
        
        current_names = self._data.columns
        drop_names = []
        for name in current_names:
            if name not in keep_names:
                drop_names.append(name)
        self.drop(drop_names)