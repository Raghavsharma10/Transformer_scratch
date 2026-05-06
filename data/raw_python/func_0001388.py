def construct_user_list(raw_users=None):
        """Construct a list of User objects from a list of dicts."""
        users = Users(oktypes=User)
        for user_dict in raw_users:
            public_keys = None
            if user_dict.get('public_keys'):
                public_keys = [PublicKey(b64encoded=x, raw=None)
                               for x in user_dict.get('public_keys')]
            users.append(User(name=user_dict.get('name'),
                              passwd=user_dict.get('passwd'),
                              uid=user_dict.get('uid'),
                              gid=user_dict.get('gid'),
                              home_dir=user_dict.get('home_dir'),
                              gecos=user_dict.get('gecos'),
                              shell=user_dict.get('shell'),
                              public_keys=public_keys,
                              sudoers_entry=user_dict.get('sudoers_entry')))
        return users