def to_dict(self):
        """ Return the user as a dict. """
        public_keys = [public_key.b64encoded for public_key in self.public_keys]
        return dict(name=self.name, passwd=self.passwd, uid=self.uid, gid=self.gid, gecos=self.gecos,
                    home_dir=self.home_dir, shell=self.shell, public_keys=public_keys)