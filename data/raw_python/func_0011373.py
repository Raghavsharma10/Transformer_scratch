def generate_subid(self, token=None):
    '''assumes a flat (file system) database, organized by experiment id, and
       subject id, with data (json) organized by subject identifier
    ''' 

    # Not headless auto-increments
    if not token:
        token = str(uuid.uuid4())

    # Headless doesn't use any folder_id, just generated token folder
    return "%s/%s" % (self.study_id, token)