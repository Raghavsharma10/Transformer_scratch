def write_members(self):
        """
        write_members
        """
        s = ""
        objattributes = list(self.m_reprdict["positional"].keys())
        objattributes.extend(list(self.m_reprdict["options"].keys()))
        objattributes.sort()

        for objattr in objattributes:
            s += 8 * " " + "self." + objattr + "="

            if objattr in self.m_reprdict["positional"]:
                td = self.m_reprdict["positional"]

                if isinstance(td[objattr], int):
                    s += "0"
                elif isinstance(td[objattr], float):
                    s += "0.0"
                elif isinstance(td[objattr], bool):
                    s += "False"
                else:
                    s += '""'
            else:
                s += "False"

            s += "\n"

        return s