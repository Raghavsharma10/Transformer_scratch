def writeShiftFile(self, filename="shifts.txt"):
        """
        Writes a shift file object to a file on disk using the convention for shift file format.
        """
        lines = ['# frame: ', self['frame'], '\n',
                 '# refimage: ', self['refimage'], '\n',
                 '# form: ', self['form'], '\n',
                 '# units: ', self['units'], '\n']

        for o in self['order']:
            ss = " "
            for shift in self[o]:
                ss += str(shift) + " "
            line = str(o) + ss + "\n"
            lines.append(line)

        fshifts= open(filename, 'w')
        fshifts.writelines(lines)
        fshifts.close()