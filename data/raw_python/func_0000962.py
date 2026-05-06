def convert_references_json(ref_content, soup=None):
    "Check for references that will not pass schema validation, fix or convert them to unknown"

    # Convert reference to unkonwn if still missing important values
    if (
        (ref_content.get("type") == "other")
        or
        (ref_content.get("type") == "book-chapter" and "editors" not in ref_content)
        or
        (ref_content.get("type") == "journal" and "articleTitle" not in ref_content)
        or
        (ref_content.get("type") in ["journal", "book-chapter"]
         and not "pages" in ref_content)
        or
        (ref_content.get("type") == "journal" and "journal" not in ref_content)
        or
        (ref_content.get("type") in ["book", "book-chapter", "report", "thesis", "software"]
         and "publisher" not in ref_content)
        or
        (ref_content.get("type") == "book" and "bookTitle" not in ref_content)
        or
        (ref_content.get("type") == "data" and "source" not in ref_content)
        or
        (ref_content.get("type") == "conference-proceeding" and "conference" not in ref_content)
       ):
        ref_content = references_json_to_unknown(ref_content, soup)

    return ref_content