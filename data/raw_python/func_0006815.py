def table_name(self):
        """Pluralises the class_name using utterly simple algo and returns as table_name"""
        if not self.class_name:
            raise ValueError
        else:
            tbl_name = ModelCompiler.convert_case(self.class_name)
        last_letter = tbl_name[-1]
        if last_letter in ("y",):
            return "{}ies".format(tbl_name[:-1])
        elif last_letter in ("s",):
            return "{}es".format(tbl_name)
        else:
            return "{}s".format(tbl_name)