def parse(self, s, tokenize=True, tags=True, chunks=True, relations=False, lemmata=False, encoding="utf-8", **kwargs):
        """ Takes a string (sentences) and returns a tagged Unicode string (TaggedString).
            Sentences in the output are separated by newlines.
            With tokenize=True, punctuation is split from words and sentences are separated by \n.
            With tags=True, part-of-speech tags are parsed (NN, VB, IN, ...).
            With chunks=True, phrase chunk tags are parsed (NP, VP, PP, PNP, ...).
            With relations=True, semantic role labels are parsed (SBJ, OBJ).
            With lemmata=True, word lemmata are parsed.
            Optional parameters are passed to
            the tokenizer, tagger, chunker, labeler and lemmatizer.
        """
        # Tokenizer.
        if tokenize is True:
            s = self.find_tokens(s, **kwargs)
        if isinstance(s, (list, tuple)):
            s = [isinstance(s, basestring) and s.split(" ") or s for s in s]
        if isinstance(s, basestring):
            s = [s.split(" ") for s in s.split("\n")]
        # Unicode.
        for i in range(len(s)):
            for j in range(len(s[i])):
                if isinstance(s[i][j], str):
                    s[i][j] = decode_string(s[i][j], encoding)
            # Tagger (required by chunker, labeler & lemmatizer).
            if tags or chunks or relations or lemmata:
                s[i] = self.find_tags(s[i], **kwargs)
            else:
                s[i] = [[w] for w in s[i]]
            # Chunker.
            if chunks or relations:
                s[i] = self.find_chunks(s[i], **kwargs)
            # Labeler.
            if relations:
                s[i] = self.find_labels(s[i], **kwargs)
            # Lemmatizer.
            if lemmata:
                s[i] = self.find_lemmata(s[i], **kwargs)
        # Slash-formatted tagged string.
        # With collapse=False (or split=True), returns raw list
        # (this output is not usable by tree.Text).
        if not kwargs.get("collapse", True) \
            or kwargs.get("split", False):
            return s
        # Construct TaggedString.format.
        # (this output is usable by tree.Text).
        format = ["word"]
        if tags:
            format.append("part-of-speech")
        if chunks:
            format.extend(("chunk", "preposition"))
        if relations:
            format.append("relation")
        if lemmata:
            format.append("lemma")
        # Collapse raw list.
        # Sentences are separated by newlines, tokens by spaces, tags by slashes.
        # Slashes in words are encoded with &slash;
        for i in range(len(s)):
            for j in range(len(s[i])):
                s[i][j][0] = s[i][j][0].replace("/", "&slash;")
                s[i][j] = "/".join(s[i][j])
            s[i] = " ".join(s[i])
        s = "\n".join(s)
        s = TaggedString(s, format, language=kwargs.get("language", self.language))
        return s