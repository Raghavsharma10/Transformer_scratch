def from_passwd(uid_min=None, uid_max=None):
        """Create collection from locally discovered data, e.g. /etc/passwd."""
        import pwd
        users = Users(oktypes=User)
        passwd_list = pwd.getpwall()
        if not uid_min:
            uid_min = UID_MIN
        if not uid_max:
            uid_max = UID_MAX
        sudoers_entries = read_sudoers()
        for pwd_entry in passwd_list:
            if uid_min <= pwd_entry.pw_uid <= uid_max:
                user = User(name=text_type(pwd_entry.pw_name),
                            passwd=text_type(pwd_entry.pw_passwd),
                            uid=pwd_entry.pw_uid,
                            gid=pwd_entry.pw_gid,
                            gecos=text_type(pwd_entry.pw_gecos),
                            home_dir=text_type(pwd_entry.pw_dir),
                            shell=text_type(pwd_entry.pw_shell),
                            public_keys=read_authorized_keys(username=pwd_entry.pw_name),
                            sudoers_entry=get_sudoers_entry(username=pwd_entry.pw_name,
                                                            sudoers_entries=sudoers_entries))
                users.append(user)
        return users