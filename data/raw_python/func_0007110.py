def _parse_text_DB(self, s):
        """Returns a dict of table interpreted from s.
        s should be Json string encoding a dict { table_name :  [fields_name,...] , [rows,... ] }"""
        dic = self.decode_json_str(s)
        new_dic = {}
        for table_name, (header, rows) in dic.items():
            newl = [{c: ligne[i]
                     for i, c in enumerate(header)} for ligne in rows]
            new_dic[table_name] = newl
        return new_dic