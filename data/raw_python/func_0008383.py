def find_lemma(self, verb):
        """ Returns the base form of the given inflected verb, using a rule-based approach.
        """
        v = verb.lower()
        # Common prefixes: be-finden and emp-finden probably inflect like finden.
        if not (v.startswith("ge") and v.endswith("t")): # Probably gerund.
            for prefix in prefixes:
                if v.startswith(prefix) and v[len(prefix):] in self.inflections:
                    return prefix + self.inflections[v[len(prefix):]]
        # Common sufixes: setze nieder => niedersetzen.
        b, suffix = " " in v and v.split()[:2] or  (v, "")
        # Infinitive -ln: trommeln.
        if b.endswith(("ln", "rn")):
            return b
        # Lemmatize regular inflections.
        for x in ("test", "est", "end", "ten", "tet", "en", "et", "te", "st", "e", "t"):
            if b.endswith(x): b = b[:-len(x)]; break
        # Subjunctive: hielte => halten, schnitte => schneiden.
        for x, y in (
          ("ieb",  "eib"), ( "ied", "eid"), ( "ief",  "auf" ), ( "ieg", "eig" ), ("iel", "alt"), 
          ("ien",  "ein"), ("iess", "ass"), (u"ieß", u"aß"  ), ( "iff", "eif" ), ("iss", "eiss"), 
          (u"iß", u"eiß"), (  "it", "eid"), ( "oss",  "iess"), (u"öss", "iess")):
            if b.endswith(x): b = b[:-len(x)] + y; break
        b = b.replace("eeiss", "eiss")
        b = b.replace("eeid", "eit")
        # Subjunctive: wechselte => wechseln
        if not b.endswith(("e", "l")) and not (b.endswith("er") and len(b) >= 3 and not b[-3] in VOWELS):
            b = b + "e"
        # abknallst != abknalln => abknallen
        if b.endswith(("hl", "ll", "ul", "eil")):
            b = b + "e"
        # Strip ge- from (likely) gerund:
        if b.startswith("ge") and v.endswith("t"):
            b = b[2:]
        # Corrections (these add about 1.5% accuracy):
        if b.endswith(("lnde", "rnde")):
            b = b[:-3]
        if b.endswith(("ae", "al", u"öe", u"üe")):
            b = b.rstrip("e") + "te"
        if b.endswith(u"äl"):
            b = b + "e"
        return suffix + b + "n"