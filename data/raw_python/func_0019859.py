def _parseAsteriskConf(self):
        """Parses Asterisk configuration file /etc/asterisk/manager.conf for
        user and password for Manager Interface. Returns True on success.
        
        @return: True if configuration file is found and parsed successfully.
        
        """
        if os.path.isfile(confFileAMI):
            try:
                fp = open(confFileAMI, 'r')
                data = fp.read()
                fp.close()
            except:
                raise IOError('Failed reading Asterisk configuration file: %s'
                    % confFileAMI)
            mobj = re.search('^\[(\w+)\]\s*\r{0,1}\nsecret\s*=\s*(\S+)\s*$', 
                             data, re.MULTILINE)
            if mobj:
                self._amiuser = mobj.group(1)
                self._amipass = mobj.group(2)
                return True
        return False