def get_aliases(self):
        """
        RETURN LIST OF {"alias":a, "index":i} PAIRS
        ALL INDEXES INCLUDED, EVEN IF NO ALIAS {"alias":Null}
        """
        for index, desc in self.get_metadata().indices.items():
            if not desc["aliases"]:
                yield wrap({"index": index})
            elif desc['aliases'][0] == index:
                Log.error("should not happen")
            else:
                for a in desc["aliases"]:
                    yield wrap({"index": index, "alias": a})