def find_lexeme(self, verb):
        """ For a regular verb (base form), returns the forms using a rule-based approach.
        """
        v = verb.lower()
        # Stem = infinitive minus -en, -ln, -rn.
        b = b0 = re.sub("en$", "", re.sub("ln$", "l", re.sub("rn$", "r", v)))
        # Split common prefixes.
        x, x1, x2 = "", "", ""
        for prefix in prefix_separable:
            if v.startswith(prefix):
                b, x = b[len(prefix):], prefix
                x1 = (" " + x).rstrip()
                x2 = x + "ge"
                break
        # Present tense 1sg and subjunctive -el: handeln => ich handle, du handlest.
        pl = b.endswith("el") and b[:-2]+"l" or b
        # Present tense 1pl -el: handeln => wir handeln
        pw = v.endswith(("ln", "rn")) and v or b+"en"
        # Present tense ending in -d or -t gets -e:
        pr = b.endswith(("d", "t")) and b+"e" or b
        # Present tense 2sg gets -st, unless stem ends with -s or -z.
        p2 = pr.endswith(("s","z")) and pr+"t" or pr+"st"
        # Present participle: spiel + -end, arbeiten + -d:
        pp = v.endswith(("en", "ln", "rn")) and v+"d" or v+"end"
        # Past tense regular:
        pt = encode_sz(pr) + "t"
        # Past participle: haushalten => hausgehalten
        ge = (v.startswith(prefix_inseparable) or b.endswith(("r","t"))) and pt or "ge"+pt
        ge = x and x+"ge"+pt or ge
        # Present subjunctive: stem + -e, -est, -en, -et:
        s1 = encode_sz(pl)
        # Past subjunctive: past (usually with Umlaut) + -e, -est, -en, -et:
        s2 = encode_sz(pt)
        # Construct the lexeme:
        lexeme = a = [
            v, 
            pl+"e"+x1, p2+x1, pr+"t"+x1, pw+x1, pr+"t"+x1, pp,             # present
            pt+"e"+x1, pt+"est"+x1, pt+"e"+x1, pt+"en"+x1, pt+"et"+x1, ge, # past
            b+"e"+x1, pr+"t"+x1, x+pw,                                     # imperative
            s1+"e"+x1, s1+"est"+x1, s1+"en"+x1, s1+"et"+x1,                # subjunctive I
            s2+"e"+x1, s2+"est"+x1, s2+"en"+x1, s2+"et"+x1                 # subjunctive II
        ]
        # Encode Eszett (ß) and attempt to retrieve from the lexicon.
        # Decode Eszett for present and imperative.
        if encode_sz(v) in self:
            a = self[encode_sz(v)]
            a = [decode_sz(v) for v in a[:7]] + a[7:13] + [decode_sz(v) for v in a[13:20]] + a[20:]
        # Since the lexicon does not contain imperative for all verbs, don't simply return it.
        # Instead, update the rule-based lexeme with inflections from the lexicon.
        return [a[i] or lexeme[i] for i in range(len(a))]