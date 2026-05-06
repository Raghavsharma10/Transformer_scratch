def set_status(self, new_status, notes=None):
        '''Save all changes and set to the given new_status'''
        self.status_id = new_status
        try:
            self.status['id'] = self.status_id
            # We don't have the id to name mapping, so blank the name
            self.status['name'] = None
        except:
            pass
        self.save(notes)