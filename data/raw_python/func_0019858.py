def _parseFreePBXconf(self):
        """Parses FreePBX configuration file /etc/amportal for user and password
        for Asterisk Manager Interface.
        
        @return: True if configuration file is found and parsed successfully.
        
        """
        amiuser = None
        amipass = None
        if os.path.isfile(confFileFreePBX):
            try:
                fp = open(confFileFreePBX, 'r')
                data = fp.read()
                fp.close()
            except:
                raise IOError('Failed reading FreePBX configuration file: %s'
                    % confFileFreePBX)
            for (key, val) in re.findall('^(AMPMGR\w+)\s*=\s*(\S+)\s*$',
                data, re.MULTILINE):
                if key == 'AMPMGRUSER':
                    amiuser = val
                elif key == 'AMPMGRPASS':
                    amipass = val
            if amiuser and amipass:
                self._amiuser = amiuser
                self._amipass = amipass
                return True
        return False