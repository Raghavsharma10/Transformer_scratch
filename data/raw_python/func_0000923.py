def refs(soup):
    """Find and return all the references"""
    tags = raw_parser.ref_list(soup)
    refs = []
    position = 1

    article_doi = doi(soup)

    for tag in tags:
        ref = {}

        ref['ref'] = ref_text(tag)

        # ref_id
        copy_attribute(tag.attrs, "id", ref)

        # article_title
        if raw_parser.article_title(tag):
            ref['article_title'] = node_text(raw_parser.article_title(tag))
            ref['full_article_title'] = node_contents_str(raw_parser.article_title(tag))

        if raw_parser.pub_id(tag, "pmid"):
            ref['pmid'] = node_contents_str(first(raw_parser.pub_id(tag, "pmid")))

        if raw_parser.pub_id(tag, "isbn"):
            ref['isbn'] = node_contents_str(first(raw_parser.pub_id(tag, "isbn")))

        if raw_parser.pub_id(tag, "doi"):
            ref['reference_id'] = node_contents_str(first(raw_parser.pub_id(tag, "doi")))
            ref['doi'] = doi_uri_to_doi(node_contents_str(first(raw_parser.pub_id(tag, "doi"))))

        uri_tag = None
        if raw_parser.ext_link(tag, "uri"):
            uri_tag = first(raw_parser.ext_link(tag, "uri"))
        elif raw_parser.uri(tag):
            uri_tag = first(raw_parser.uri(tag))
        if uri_tag:
            set_if_value(ref, "uri", uri_tag.get('xlink:href'))
            set_if_value(ref, "uri_text", node_contents_str(uri_tag))
        # look for a pub-id tag if no uri yet
        if not ref.get('uri') and raw_parser.pub_id(tag, "archive"):
            pub_id_tag = first(raw_parser.pub_id(tag, pub_id_type="archive"))
            set_if_value(ref, "uri", pub_id_tag.get('xlink:href'))

        # accession, could be in either of two tags
        set_if_value(ref, "accession", node_contents_str(first(raw_parser.object_id(tag, "art-access-id"))))
        if not ref.get('accession'):
            set_if_value(ref, "accession", node_contents_str(first(raw_parser.pub_id(tag, pub_id_type="accession"))))
        if not ref.get('accession'):
            set_if_value(ref, "accession", node_contents_str(first(raw_parser.pub_id(tag, pub_id_type="archive"))))

        if(raw_parser.year(tag)):
            set_if_value(ref, "year", node_text(raw_parser.year(tag)))
            set_if_value(ref, "year-iso-8601-date", raw_parser.year(tag).get('iso-8601-date'))

        if(raw_parser.date_in_citation(tag)):
            set_if_value(ref, "date-in-citation", node_text(first(raw_parser.date_in_citation(tag))))
            set_if_value(ref, "iso-8601-date", first(raw_parser.date_in_citation(tag)).get('iso-8601-date'))

        if(raw_parser.patent(tag)):
            set_if_value(ref, "patent", node_text(first(raw_parser.patent(tag))))
            set_if_value(ref, "country", first(raw_parser.patent(tag)).get('country'))

        set_if_value(ref, "source", node_text(first(raw_parser.source(tag))))
        set_if_value(ref, "elocation-id", node_text(first(raw_parser.elocation_id(tag))))
        if raw_parser.element_citation(tag):
            copy_attribute(first(raw_parser.element_citation(tag)).attrs, "publication-type", ref)
        if "publication-type" not in ref and raw_parser.mixed_citations(tag):
            copy_attribute(first(raw_parser.mixed_citations(tag)).attrs, "publication-type", ref)

        # authors
        person_group = raw_parser.person_group(tag)
        authors = []

        for group in person_group:

            author_type = None
            if "person-group-type" in group.attrs:
                author_type = group["person-group-type"]

            # Read name or collab tag in the order they are listed
            for name_or_collab_tag in extract_nodes(group, ["name", "string-name", "collab"]):
                author = {}

                # Shared tag attribute
                set_if_value(author, "group-type", author_type)

                # name tag attributes
                if name_or_collab_tag.name in ["name", "string-name"]:
                    set_if_value(author, "surname", node_text(first(raw_parser.surname(name_or_collab_tag))))
                    set_if_value(author, "given-names", node_text(first(raw_parser.given_names(name_or_collab_tag))))
                    set_if_value(author, "suffix", node_text(first(raw_parser.suffix(name_or_collab_tag))))

                # collab tag attribute
                if name_or_collab_tag.name == "collab":
                    set_if_value(author, "collab", node_contents_str(name_or_collab_tag))

                if len(author) > 0:
                    authors.append(author)

            # etal for the person group
            if first(raw_parser.etal(group)):
                author = {}
                author['etal'] = True
                set_if_value(author, "group-type", author_type)
                authors.append(author)

        # Check for collab tag not wrapped in a person-group for backwards compatibility
        if len(person_group) == 0:
            collab_tags = raw_parser.collab(tag)
            for collab_tag in collab_tags:
                author = {}
                set_if_value(author, "group-type", "author")
                set_if_value(author, "collab", node_contents_str(collab_tag))

                if len(author) > 0:
                    authors.append(author)

        if len(authors) > 0:
            ref['authors'] = authors

        set_if_value(ref, "volume", node_text(first(raw_parser.volume(tag))))
        set_if_value(ref, "issue", node_text(first(raw_parser.issue(tag))))
        set_if_value(ref, "fpage", node_text(first(raw_parser.fpage(tag))))
        set_if_value(ref, "lpage", node_text(first(raw_parser.lpage(tag))))
        set_if_value(ref, "collab", node_text(first(raw_parser.collab(tag))))
        set_if_value(ref, "publisher_loc", node_text(first(raw_parser.publisher_loc(tag))))
        set_if_value(ref, "publisher_name", node_text(first(raw_parser.publisher_name(tag))))
        set_if_value(ref, "edition", node_contents_str(first(raw_parser.edition(tag))))
        set_if_value(ref, "version", node_contents_str(first(raw_parser.version(tag))))
        set_if_value(ref, "chapter-title", node_contents_str(first(raw_parser.chapter_title(tag))))
        set_if_value(ref, "comment", node_text(first(raw_parser.comment(tag))))
        set_if_value(ref, "data-title", node_contents_str(first(raw_parser.data_title(tag))))
        set_if_value(ref, "conf-name", node_text(first(raw_parser.conf_name(tag))))

        # If not empty, add position value, append, then increment the position counter
        if(len(ref) > 0):
            ref['article_doi'] = article_doi

            ref['position'] = position

            refs.append(ref)
            position += 1

    return refs