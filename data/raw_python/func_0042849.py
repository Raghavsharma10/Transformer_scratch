def new_leaves(self, column_name):
        """
        :param column_name:
        :return: ALL COLUMNS THAT START WITH column_name, INCLUDING DEEP COLUMNS
        """
        column_name = unnest_path(column_name)
        columns = self.columns
        all_paths = self.snowflake.sorted_query_paths

        output = {}
        for c in columns:
            if c.name == "_id" and column_name != "_id":
                continue
            if c.jx_type in OBJECTS:
                continue
            if c.cardinality == 0:
                continue
            for path in all_paths:
                if not startswith_field(unnest_path(relative_field(c.name, path)), column_name):
                    continue
                existing = output.get(path)
                if not existing:
                    output[path] = [c]
                    continue
                if len(path) > len(c.nested_path[0]):
                    continue
                if any("." + t + "." in c.es_column for t in (STRING_TYPE, NUMBER_TYPE, BOOLEAN_TYPE)):
                    # ELASTICSEARCH field TYPES ARE NOT ALLOWED
                    continue
                # ONLY THE DEEPEST COLUMN WILL BE CHOSEN
                output[path].append(c)
        return set(output.values())