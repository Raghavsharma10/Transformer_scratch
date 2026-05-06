def start_group(self, scol, typ):
        """Start a new group"""
        return Group(parent=self, level=scol, typ=typ)