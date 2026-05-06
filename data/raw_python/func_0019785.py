def _formatConfig(self, conf_dict):
        """Formats configuration directory from Munin Graph and returns 
        multi-line value entries for the plugin config cycle.
        
        @param conf_dict: Configuration directory. 
        @return:          Multi-line text.
        
        """
        confs = []
        graph_dict = conf_dict['graph']
        field_list = conf_dict['fields']
        
        # Order and format Graph Attributes
        title = graph_dict.get('title')
        if title is not None:
            if self.isMultiInstance and self._instanceLabel is not None:
                if self._instanceLabelType == 'suffix':
                    confs.append("graph_%s %s - %s" % ('title', 
                                                       title, 
                                                       self._instanceLabel,))
                elif self._instanceLabelType == 'prefix':
                    confs.append("graph_%s %s - %s" % ('title', 
                                                       self._instanceLabel,
                                                       title,))
            else:
                confs.append("graph_%s %s" % ('title', title))
        for key in ('category', 'vlabel', 'info', 'args', 'period', 
                    'scale', 'total', 'order', 'printf', 'width', 'height'):
            val = graph_dict.get(key)
            if val is not None:
                if isinstance(val, bool):
                    if val:
                        val = "yes"
                    else:
                        val = "no"
                confs.append("graph_%s %s" % (key, val))

        # Order and Format Field Attributes
        for (field_name, field_attrs) in field_list:
            for key in ('label', 'type', 'draw', 'info', 'extinfo', 'colour',
                        'negative', 'graph', 'min', 'max', 'cdef', 
                        'line', 'warning', 'critical'):
                val = field_attrs.get(key)
                if val is not None:
                    if isinstance(val, bool):
                        if val:
                            val = "yes"
                        else:
                            val = "no"
                    confs.append("%s.%s %s" % (field_name, key, val))
        return "\n".join(confs)