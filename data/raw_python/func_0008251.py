def loadJSON(self, json_string):
        """
        Sets the values of the run parameters given an JSON string
        """
        g = get_root(self).globals
        user = json.loads(json_string)['user']

        def setField(widget, field):
            val = user.get(field)
            if val is not None:
                widget.set(val)

        setField(self.prog_ob.obid, 'OB')
        setField(self.target, 'target')
        setField(self.prog_ob.progid, 'ID')
        setField(self.pi, 'PI')
        setField(self.observers, 'Observers')
        setField(self.comment, 'comment')
        setField(self.filter, 'filters')
        setField(g.observe.rtype, 'flags')