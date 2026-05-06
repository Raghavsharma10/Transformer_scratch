def addField(self, name, label, type=None,  draw=None, info=None, #@ReservedAssignment
                 extinfo=None, colour=None, negative=None, graph=None, 
                 min=None, max=None, cdef=None, line=None, #@ReservedAssignment
                 warning=None, critical=None):
        """Add field to Munin Graph
        
            @param name:     Field Name
            @param label:    Field Label
            @param type:     Stat Type:
                             'COUNTER' / 'ABSOLUTE' / 'DERIVE' / 'GAUGE'
            @param draw:     Graph Type:
                             'AREA' / 'LINE{1,2,3}' / 
                             'STACK' / 'LINESTACK{1,2,3}' / 'AREASTACK'
            @param info:     Detailed Field Info
            @param extinfo:  Extended Field Info
            @param colour:   Field Colour
            @param negative: Mirror Value
            @param graph:    Draw on Graph - True / False (Default: True)
            @param min:      Minimum Valid Value
            @param max:      Maximum Valid Value
            @param cdef:     CDEF
            @param line:     Adds horizontal line at value defined for field. 
            @param warning:  Warning Value
            @param critical: Critical Value
            
        """
        if self._autoFixNames:
            name = self._fixName(name)
            if negative is not None:
                negative = self._fixName(negative)
        self._fieldAttrDict[name] = dict(((k,v) for (k,v) in locals().iteritems()
                                         if (v is not None
                                             and k not in ('self',))))
        self._fieldNameList.append(name)