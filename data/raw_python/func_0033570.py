def get_assets(self):
        '''
        Return a flat list of absolute paths to all assets required by this
        viewer
        '''
        return sum([
            [self.prefix_asset(viewer, relpath) for relpath in viewer.assets]
            for viewer in self.viewers
        ], [])