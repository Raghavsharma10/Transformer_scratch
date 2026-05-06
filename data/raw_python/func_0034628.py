def to_safe(self, word):
        ''' Converts 'bad' characters in a string to underscores so they can be used as Ansible groups '''
        regex = "[^A-Za-z0-9\_"
        if not self.replace_dash_in_groups:
            regex += "\-"
        return re.sub(regex + "]", "_", word)