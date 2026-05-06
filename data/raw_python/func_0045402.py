def get_subclass(self):
        """
        get_subclass
        """
        strbldr = """
            class IArguments(Arguments):
                \"\"\"
                IArguments
                \"\"\"
                def __init__(self, doc=None, validateschema=None, argvalue=None, yamlstr=None, yamlfile=None, parse_arguments=True, persistoption=False, alwaysfullhelp=False, version=None, parent=None):
                    \"\"\"
                    @type doc: str, None
                    @type validateschema: Schema, None
                    @type yamlfile: str, None
                    @type yamlstr: str, None
                    @type parse_arguments: bool
                    @type argvalue: str, None
                    @return: None
                    \"\"\"
        """
        strbldr = remove_extra_indentation(strbldr)
        strbldr += "\n"
        self.set_reprdict_from_attributes()
        strbldr += self.write_members()
        strbldr += 8 * " " + "super().__init__(doc, validateschema, argvalue, yamlstr, yamlfile, parse_arguments, persistoption, alwaysfullhelp, version, parent)\n\n"
        strbldr2 = """
            class IArguments(Arguments):
                \"\"\"
                IArguments
                \"\"\"
                def __init__(self, doc):
                    \"\"\"
                    __init__
                    \"\"\"
        """
        strbldr2 = remove_extra_indentation(strbldr2)
        strbldr2 += "\n"
        strbldr2 += self.write_members()
        strbldr2 += 8 * " " + "super().__init__(doc)\n\n"
        return strbldr2