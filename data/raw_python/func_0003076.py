def to_escpos(self):
        """ converts the current style to an escpos command string """
        cmd = ''
        ordered_cmds = self.cmds.keys()
        ordered_cmds.sort(lambda x,y: cmp(self.cmds[x]['_order'], self.cmds[y]['_order']))
        for style in ordered_cmds:
            cmd += self.cmds[style][self.get(style)]
        return cmd