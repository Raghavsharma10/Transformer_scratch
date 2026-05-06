def _formatVals(self, val_list):
        """Formats value list from Munin Graph and returns multi-line value
        entries for the plugin fetch cycle.
        
        @param val_list: List of name-value pairs. 
        @return:         Multi-line text.
        
        """
        vals = []
        for (name, val) in val_list:
            if val is not None:
                if isinstance(val, float):
                    vals.append("%s.value %f" % (name, val))
                else:
                    vals.append("%s.value %s" % (name, val))
            else:
                vals.append("%s.value U" % (name,))
        return "\n".join(vals)