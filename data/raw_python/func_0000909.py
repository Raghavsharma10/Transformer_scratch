def abstracts(soup):
    """
    Find the article abstract and format it
    """

    abstracts = []

    abstract_tags = raw_parser.abstract(soup)

    for tag in abstract_tags:
        abstract = {}

        abstract["abstract_type"] = tag.get("abstract-type")
        title_tag = raw_parser.title(tag)
        if title_tag:
            abstract["title"] = node_text(title_tag)

        abstract["content"] = None
        if raw_parser.paragraph(tag):
            abstract["content"] = ""
            abstract["full_content"] = ""

            good_paragraphs = remove_doi_paragraph(raw_parser.paragraph(tag))

            # Plain text content
            glue = ""
            for p_tag in good_paragraphs:
                abstract["content"] += glue + node_text(p_tag)
                glue = " "

            # Content including markup tags
            # When more than one paragraph, wrap each in a <p> tag
            for p_tag in good_paragraphs:
                abstract["full_content"] += '<p>' + node_contents_str(p_tag) + '</p>'

        abstracts.append(abstract)

    return abstracts