def is_draft_version(self):
        """ Return if this version is the draft version of a layer """
        pub_ver = getattr(self, 'published_version', None)
        latest_ver = getattr(self, 'latest_version', None)
        this_ver = getattr(self, 'this_version', None)
        return this_ver and latest_ver and (this_ver == latest_ver) and (latest_ver != pub_ver)