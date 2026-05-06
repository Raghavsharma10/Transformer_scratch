def fromstring(cls, s, *args, **kwargs):
        """ Returns a new Pattern from the given string.
            Constraints are separated by a space.
            If a constraint contains a space, it must be wrapped in [].
        """
        s = s.replace("\(", "&lparen;")
        s = s.replace("\)", "&rparen;")
        s = s.replace("\[", "&lbrack;")
        s = s.replace("\]", "&rbrack;")
        s = s.replace("\{", "&lcurly;")
        s = s.replace("\}", "&rcurly;")
        p = []
        i = 0
        for m in re.finditer(r"\[.*?\]|\(.*?\)", s):
            # Spaces in a range encapsulated in square brackets are encoded.
            # "[Windows Vista]" is one range, don't split on space.
            p.append(s[i:m.start()])
            p.append(s[m.start():m.end()].replace(" ", "&space;")); i=m.end()
        p.append(s[i:])
        s = "".join(p) 
        s = s.replace("][", "] [")
        s = s.replace(")(", ") (")
        s = s.replace("\|", "&vdash;")
        s = re.sub(r"\s+\|\s+", "|", s)  
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\{\s+", "{", s)
        s = re.sub(r"\s+\}", "}", s)
        s = s.split(" ")
        s = [v.replace("&space;"," ") for v in s]
        P = cls([], *args, **kwargs)
        G, O, i = [], [], 0
        for s in s:
            constraint = Constraint.fromstring(s.strip("{}"), taxonomy=kwargs.get("taxonomy", TAXONOMY))
            constraint.index = len(P.sequence)
            P.sequence.append(constraint)
            # Push a new group on the stack if string starts with "{".
            # Parse constraint from string, add it to all open groups.
            # Pop latest group from stack if string ends with "}".
            # Insert groups in opened-first order (i).
            while s.startswith("{"):
                s = s[1:]
                G.append((i, [])); i+=1
                O.append([])
            for g in G:
                g[1].append(constraint)
            while s.endswith("}"):
                s = s[:-1]
                if G: O[G[-1][0]] = G[-1][1]; G.pop()
        P.groups = [g for g in O if g]
        return P