def get_canonical_index(self, alias):
        """
        RETURN THE INDEX USED BY THIS alias
        THIS IS ACCORDING TO THE STRICT LIFECYCLE RULES:
        THERE IS ONLY ONE INDEX WITH AN ALIAS
        """
        output = jx.sort(set(
            i
            for ai in self.get_aliases()
            for a, i in [(ai.alias, ai.index)]
            if a == alias or i == alias or (re.match(re.escape(alias) + "\\d{8}_\\d{6}", i) and i != alias)
        ))

        if len(output) > 1:
            Log.error("only one index with given alias==\"{{alias}}\" expected", alias=alias)

        if not output:
            return Null

        return output.last()