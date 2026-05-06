def _convert_clause(self, clause):
        """
        JSON QUERY EXPRESSIONS HAVE MANY CLAUSES WITH SIMILAR COLUMN DELCARATIONS
        """
        clause = wrap(clause)

        if clause == None:
            return None
        elif is_data(clause):
            return set_default({"value": self.convert(clause.value)}, clause)
        else:
            return [set_default({"value": self.convert(c.value)}, c) for c in clause]