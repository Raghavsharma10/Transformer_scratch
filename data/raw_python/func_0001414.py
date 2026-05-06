def read(self, doc=None):
        ''' Read tagged doc from mutliple files (sents, tokens, concepts, links, tags) '''
        if not self.sent_stream:
            raise Exception("There is no sentence data stream available")
        if doc is None:
            doc = Document(name=self.doc_name, path=self.doc_path)
        for row in self.sent_reader():
            if len(row) == 2:
                sid, text = row
                doc.new_sent(text.strip(), ID=sid)
            elif len(row) == 4:
                sid, text, flag, comment = row
                sent = doc.new_sent(text.strip(), ID=sid)
                sent.flag = flag
                sent.comment = comment
        # Read tokens if available
        if self.token_stream:
            # read all tokens first
            sent_tokens_map = dd(list)
            for token_row in self.token_reader():
                if len(token_row) == 6:
                    sid, wid, token, lemma, pos, comment = token_row
                else:
                    sid, wid, token, lemma, pos = token_row
                    comment = ''
                sid = int(sid)
                sent_tokens_map[sid].append((token, lemma, pos.strip(), wid, comment))
                # TODO: verify wid?
            # import tokens
            for sent in doc:
                sent_tokens = sent_tokens_map[sent.ID]
                sent.import_tokens([x[0] for x in sent_tokens])
                for ((tk, lemma, pos, wid, comment), token) in zip(sent_tokens, sent.tokens):
                    token.pos = pos
                    token.lemma = lemma
                    token.comment = comment
            # only read concepts if tokens are available
            if self.concept_stream:
                # read concepts
                for concept_row in self.concept_reader():
                    if len(concept_row) == 5:
                        sid, cid, clemma, tag, comment = concept_row
                    else:
                        sid, cid, clemma, tag = concept_row
                        comment = ''
                    cid = int(cid)
                    doc.get(sid).new_concept(tag.strip(), clemma=clemma, cidx=cid, comment=comment)
                # only read concept-token links if tokens and concepts are available
                for sid, cid, wid in self.link_reader():
                    sent = doc.get(sid)
                    cid = int(cid)
                    wid = int(wid.strip())
                    sent.concept(cid).add_token(sent[wid])
        # read sentence level tags
        if self.tag_stream:
            for row in self.tag_reader():
                if len(row) == 5:
                    sid, cfrom, cto, label, tagtype = row
                    wid = None
                if len(row) == 6:
                    sid, cfrom, cto, label, tagtype, wid = row
                if cfrom:
                    cfrom = int(cfrom)
                if cto:
                    cto = int(cto)
                if wid is None or wid == '':
                    doc.get(sid).new_tag(label, cfrom, cto, tagtype=tagtype)
                else:
                    doc.get(sid)[int(wid)].new_tag(label, cfrom, cto, tagtype=tagtype)
        return doc