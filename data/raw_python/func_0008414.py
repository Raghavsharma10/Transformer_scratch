def fromstring(cls, s, **kwargs):
        """ Returns a new Constraint from the given string.
            Uppercase words indicate either a tag ("NN", "JJ", "VP")
            or a taxonomy term (e.g., "PRODUCT", "PERSON").
            Syntax:
            ( defines an optional constraint, e.g., "(JJ)".
            [ defines a constraint with spaces, e.g., "[Mac OS X | Windows Vista]".
            _ is converted to spaces, e.g., "Windows_Vista".
            | separates different options, e.g., "ADJP|ADVP".
            ! can be used as a word prefix to disallow it.
            * can be used as a wildcard character, e.g., "soft*|JJ*".
            ? as a suffix defines a constraint that is optional, e.g., "JJ?".
            + as a suffix defines a constraint that can span multiple words, e.g., "JJ+".
            ^ as a prefix defines a constraint that can only match the first word.
            These characters need to be escaped if used as content: "\(".
        """
        C = cls(**kwargs)
        s = s.strip()
        s = s.strip("{}")
        s = s.strip()
        for i in range(3):
            # Wrapping order of control characters is ignored:
            # (NN+) == (NN)+ == NN?+ == NN+? == [NN+?] == [NN]+?
            if s.startswith("^"):
                s = s[1:  ]; C.first = True
            if s.endswith("+") and not s.endswith("\+"):
                s = s[0:-1]; C.multiple = True
            if s.endswith("?") and not s.endswith("\?"):
                s = s[0:-1]; C.optional = True
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]; C.optional = True
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
        s = re.sub(r"^\\\^", "^", s)
        s = re.sub(r"\\\+$", "+", s)
        s = s.replace("\_", "&uscore;")
        s = s.replace("_"," ")
        s = s.replace("&uscore;", "_")
        s = s.replace("&lparen;", "(")
        s = s.replace("&rparen;", ")")
        s = s.replace("&lbrack;", "[")
        s = s.replace("&rbrack;", "]")
        s = s.replace("&lcurly;", "{")
        s = s.replace("&rcurly;", "}")
        s = s.replace("\(", "(")
        s = s.replace("\)", ")") 
        s = s.replace("\[", "[")
        s = s.replace("\]", "]") 
        s = s.replace("\{", "{")
        s = s.replace("\}", "}") 
        s = s.replace("\*", "*")
        s = s.replace("\?", "?")    
        s = s.replace("\+", "+")
        s = s.replace("\^", "^")
        s = s.replace("\|", "&vdash;")
        s = s.split("|")
        s = [v.replace("&vdash;", "|").strip() for v in s]
        for v in s:
            C._append(v)
        return C