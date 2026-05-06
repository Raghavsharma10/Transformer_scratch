def _parse_relation(self, tag):
        """ Parses the chunk tag, role and relation id from the token relation tag.
            - VP                => VP, [], []
            - VP-1              => VP, [1], [None]
            - ADJP-PRD          => ADJP, [None], [PRD]
            - NP-SBJ-1          => NP, [1], [SBJ]
            - NP-OBJ-1*NP-OBJ-2 => NP, [1,2], [OBJ,OBJ]
            - NP-SBJ;NP-OBJ-1   => NP, [1,1], [SBJ,OBJ]
        """
        chunk, relation, role = None, [], []
        if ";" in tag:
            # NP-SBJ;NP-OBJ-1 => 1 relates to both SBJ and OBJ.
            id = tag.split("*")[0][-2:]
            id = id if id.startswith("-") else ""
            tag = tag.replace(";", id + "*")
        if "*" in tag:
            tag = tag.split("*")
        else:
            tag = [tag]
        for s in tag:
            s = s.split("-")
            n = len(s)
            if n == 1: 
                chunk = s[0]
            if n == 2: 
                chunk = s[0]; relation.append(s[1]); role.append(None)
            if n >= 3: 
                chunk = s[0]; relation.append(s[2]); role.append(s[1])
            if n > 1:
                id = relation[-1]
                if id.isdigit():
                    relation[-1] = int(id)
                else:
                    # Correct "ADJP-PRD":
                    # (ADJP, [PRD], [None]) => (ADJP, [None], [PRD])
                    relation[-1], role[-1] = None, id
        return chunk, relation, role