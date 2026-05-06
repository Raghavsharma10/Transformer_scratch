def is_published_version(self):
        """ Return if this version is the published version of a layer """
        pub_ver = getattr(self, 'published_version', None)
        this_ver = getattr(self, 'this_version', None)
        return this_ver and pub_ver and (this_ver == pub_ver)