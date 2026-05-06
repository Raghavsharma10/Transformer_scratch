def save_data(self, session, exp_id, content):
    '''save data will obtain the current subid from the session, and save it
       depending on the database type. Currently we just support flat files'''

    subid = session.get('subid')

    # We only attempt save if there is a subject id, set at start
    data_file = None
    if subid is not None:

        data_base = "%s/%s" %(self.data_base, subid)

        # If not running in headless, ensure path exists
        if not self.headless and not os.path.exists(data_base):
            mkdir_p(data_base)

        # Conditions for saving:
        do_save = False

        # If headless with token pre-generated OR not headless
        if self.headless and os.path.exists(data_base) or not self.headless:
            do_save = True
        if data_base.endswith(('revoked','finished')):
            do_save = False  

        # If headless with token pre-generated OR not headless
        if do_save is True:
            data_file = "%s/%s-results.json" %(data_base, exp_id)
            if os.path.exists(data_file):
                self.logger.warning('%s exists, and is being overwritten.' %data_file)
            write_json(content, data_file)

    return data_file