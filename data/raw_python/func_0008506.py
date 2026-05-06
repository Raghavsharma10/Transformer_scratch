def parse_xml(sentence, tab="\t", id=""):
    """ Returns the given Sentence object as an XML-string (plain bytestring, UTF-8 encoded).
        The tab delimiter is used as indendation for nested elements.
        The id can be used as a unique identifier per sentence for chunk id's and anchors.
        For example: "I eat pizza with a fork." =>
        
        <sentence token="word, part-of-speech, chunk, preposition, relation, anchor, lemma" language="en">
            <chunk type="NP" relation="SBJ" of="1">
                <word type="PRP" lemma="i">I</word>
            </chunk>
            <chunk type="VP" relation="VP" id="1" anchor="A1">
                <word type="VBP" lemma="eat">eat</word>
            </chunk>
            <chunk type="NP" relation="OBJ" of="1">
                <word type="NN" lemma="pizza">pizza</word>
            </chunk>
            <chunk type="PNP" of="A1">
                <chunk type="PP">
                    <word type="IN" lemma="with">with</word>
                </chunk>
                <chunk type="NP">
                    <word type="DT" lemma="a">a</word>
                    <word type="NN" lemma="fork">fork</word>
                </chunk>
            </chunk>
            <chink>
                <word type="." lemma=".">.</word>
            </chink>
        </sentence>
    """
    uid  = lambda *parts: "".join([str(id), _UID_SEPARATOR ]+[str(x) for x in parts]).lstrip(_UID_SEPARATOR)
    push = lambda indent: indent+tab         # push() increases the indentation.
    pop  = lambda indent: indent[:-len(tab)] # pop() decreases the indentation.
    indent = tab
    xml = []
    # Start the sentence element:
    # <sentence token="word, part-of-speech, chunk, preposition, relation, anchor, lemma">
    xml.append('<%s%s %s="%s" %s="%s">' % (
        XML_SENTENCE,
        XML_ID and " %s=\"%s\"" % (XML_ID, str(id)) or "",
        XML_TOKEN, ", ".join(sentence.token),
        XML_LANGUAGE, sentence.language
    ))
    # Collect chunks that are PNP anchors and assign id.
    anchors = {}
    for chunk in sentence.chunks:
        if chunk.attachments:
            anchors[chunk.start] = len(anchors) + 1
    # Traverse all words in the sentence.
    for word in sentence.words:
        chunk = word.chunk
        pnp   = word.chunk and word.chunk.pnp or None
        # Start the PNP element if the chunk is the first chunk in PNP:
        # <chunk type="PNP" of="A1">
        if pnp and pnp.start == chunk.start:
            a = pnp.anchor and ' %s="%s"' % (XML_OF, uid("A", anchors.get(pnp.anchor.start, ""))) or ""
            xml.append(indent + '<%s %s="PNP"%s>' % (XML_CHUNK, XML_TYPE, a))
            indent = push(indent)
        # Start the chunk element if the word is the first word in the chunk:
        # <chunk type="VP" relation="VP" id="1" anchor="A1">
        if chunk and chunk.start == word.index:
            if chunk.relations:
                # Create the shortest possible attribute values for multiple relations, 
                # e.g., [(1,"OBJ"),(2,"OBJ")]) => relation="OBJ" id="1|2"
                r1 = unzip(0, chunk.relations) # Relation id's.
                r2 = unzip(1, chunk.relations) # Relation roles.
                r1 = [x is None and "-" or uid(x) for x in r1]
                r2 = [x is None and "-" or x for x in r2]
                r1 = not len(unique(r1)) == 1 and "|".join(r1) or (r1+[None])[0]
                r2 = not len(unique(r2)) == 1 and "|".join(r2) or (r2+[None])[0]
            xml.append(indent + '<%s%s%s%s%s%s>' % (
                XML_CHUNK,
                chunk.type and ' %s="%s"' % (XML_TYPE, chunk.type) or "",
                chunk.relations and chunk.role != None and ' %s="%s"' % (XML_RELATION, r2) or "",
                chunk.relation  and chunk.type == "VP" and ' %s="%s"' % (XML_ID, uid(chunk.relation)) or "",
                chunk.relation  and chunk.type != "VP" and ' %s="%s"' % (XML_OF, r1) or "",
                chunk.attachments and ' %s="%s"' % (XML_ANCHOR, uid("A",anchors[chunk.start])) or ""
            ))
            indent = push(indent)
        # Words outside of a chunk are wrapped in a <chink> tag:
        # <chink>
        if not chunk:
            xml.append(indent + '<%s>' % XML_CHINK)
            indent = push(indent)
        # Add the word element:
        # <word type="VBP" lemma="eat">eat</word>
        xml.append(indent + '<%s%s%s%s>%s</%s>' % (
            XML_WORD,
            word.type and ' %s="%s"' % (XML_TYPE, xml_encode(word.type)) or '',
            word.lemma and ' %s="%s"' % (XML_LEMMA, xml_encode(word.lemma)) or '',
            (" "+" ".join(['%s="%s"' % (k,v) for k,v in word.custom_tags.items() if v != None])).rstrip(),
            xml_encode(unicode(word)),
            XML_WORD
        ))
        if not chunk:
            # Close the <chink> element if outside of a chunk.
            indent = pop(indent); xml.append(indent + "</%s>" % XML_CHINK)
        if chunk and chunk.stop-1 == word.index:
            # Close the <chunk> element if this is the last word in the chunk.
            indent = pop(indent); xml.append(indent + "</%s>" % XML_CHUNK)
        if pnp and pnp.stop-1 == word.index:
            # Close the PNP element if this is the last word in the PNP.
            indent = pop(indent); xml.append(indent + "</%s>" % XML_CHUNK)
    xml.append("</%s>" % XML_SENTENCE)
    # Return as a plain str.
    return "\n".join(xml).encode("utf-8")