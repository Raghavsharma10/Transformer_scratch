def delete_all_but(self, prefix, name):
        """
        :param prefix: INDEX MUST HAVE THIS AS A PREFIX AND THE REMAINDER MUST BE DATE_TIME
        :param name: INDEX WITH THIS NAME IS NOT DELETED
        :return:
        """
        if prefix == name:
            Log.note("{{index_name}} will not be deleted", {"index_name": prefix})
        for a in self.get_aliases():
            # MATCH <prefix>YYMMDD_HHMMSS FORMAT
            if re.match(re.escape(prefix) + "\\d{8}_\\d{6}", a.index) and a.index != name:
                self.delete_index(a.index)