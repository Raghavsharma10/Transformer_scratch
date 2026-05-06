def _parseSections(self, data):
        """Parse data and separate sections. Returns dictionary that maps 
        section name to section data.
        
        @param data: Multiline data.
        @return:     Dictionary that maps section names to section data.
        
        """
        section_dict = {}
        lines = data.splitlines()
        idx = 0
        numlines = len(lines)
        section = None
        while idx < numlines:
            line = lines[idx]
            idx += 1
            mobj = re.match('^(\w[\w\s\(\)]+[\w\)])\s*:$', line)
            if mobj:
                section = mobj.group(1)
                section_dict[section] = []
            else:
                mobj = re.match('(\t|\s)\s*(\w.*)$', line)
                if mobj:
                    section_dict[section].append(mobj.group(2))
                else:
                    mobj = re.match('^(\w[\w\s\(\)]+[\w\)])\s*:\s*(\S.*)$', line)
                    if mobj:
                        section = None
                        if not section_dict.has_key(section):
                            section_dict[section] = []
                        section_dict[section].append(line)
                    else:
                        if not section_dict.has_key('PARSEERROR'):
                            section_dict['PARSEERROR'] = []
                        section_dict['PARSEERROR'].append(line)   
        return section_dict