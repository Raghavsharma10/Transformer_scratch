def name(self):
        '''Returns the name of this template (if created from a file) or "string" if not'''
        if self.mako_template.filename:
            return os.path.basename(self.mako_template.filename)
        return 'string'