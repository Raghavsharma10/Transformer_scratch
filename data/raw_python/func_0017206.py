def get_ascii(self, show_internal=True, compact=False, attributes=None):
        """
        Returns a string containing an ascii drawing of the tree.
        
        Parameters:
        -----------
        show_internal: 
            include internal edge names.
        compact: 
            use exactly one line per tip.
        attributes: 
            A list of node attributes to shown in the ASCII representation.
        """
        (lines, mid) = self._asciiArt(show_internal=show_internal,
                                      compact=compact, 
                                      attributes=attributes)
        return '\n'+'\n'.join(lines)