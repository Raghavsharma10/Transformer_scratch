def _parse_token(word, chunk="O", pnp="O", relation="O", anchor="O", 
                 format=[WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA]):
    """ Returns a list of token tags parsed from the given <word> element.
        Tags that are not attributes in a <word> (e.g., relation) can be given as parameters.
    """
    tags = []
    for tag in format:
        if   tag == WORD   : tags.append(xml_decode(word.value))
        elif tag == POS    : tags.append(xml_decode(word.get(XML_TYPE, "O")))
        elif tag == CHUNK  : tags.append(chunk)
        elif tag == PNP    : tags.append(pnp)
        elif tag == REL    : tags.append(relation)
        elif tag == ANCHOR : tags.append(anchor)
        elif tag == LEMMA  : tags.append(xml_decode(word.get(XML_LEMMA, "")))
        else:
            # Custom tags when the parser has been extended, see also Word.custom_tags{}.
            tags.append(xml_decode(word.get(tag, "O")))
    return tags