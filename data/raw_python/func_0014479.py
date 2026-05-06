def save_host_keys(self, filename):
        """
        Save the host keys back to a file.  Only the host keys loaded with
        L{load_host_keys} (plus any added directly) will be saved -- not any
        host keys loaded with L{load_system_host_keys}.

        @param filename: the filename to save to
        @type filename: str

        @raise IOError: if the file could not be written
        """
        f = open(filename, 'w')
        f.write('# SSH host keys collected by ssh\n')
        for hostname, keys in self._host_keys.iteritems():
            for keytype, key in keys.iteritems():
                f.write('%s %s %s\n' % (hostname, keytype, key.get_base64()))
        f.close()