def _parseStats(self, lines, parse_slabs = False):
        """Parse stats output from memcached and return dictionary of stats-
        
        @param lines:       Array of lines of input text.
        @param parse_slabs: Parse slab stats if True.
        @return:            Stats dictionary.
        
        """
        info_dict = {}
        info_dict['slabs'] = {}
        for line in lines:
            mobj = re.match('^STAT\s(\w+)\s(\S+)$',  line)
            if mobj:
                info_dict[mobj.group(1)] = util.parse_value(mobj.group(2), True)
                continue
            elif parse_slabs:
                mobj = re.match('STAT\s(\w+:)?(\d+):(\w+)\s(\S+)$',  line)
                if mobj:
                    (slab, key, val) = mobj.groups()[-3:]      
                    if not info_dict['slabs'].has_key(slab):
                        info_dict['slabs'][slab] = {}
                    info_dict['slabs'][slab][key] = util.parse_value(val, True)
        return info_dict