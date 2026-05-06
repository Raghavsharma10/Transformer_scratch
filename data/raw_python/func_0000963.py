def references_json_unknown_details(ref_content, soup=None):
    "Extract detail value for references of type unknown"
    details = ""

    # Try adding pages values first
    if "pages" in ref_content:
        if "range" in ref_content["pages"]:
            details += ref_content["pages"]["range"]
        else:
            details += ref_content["pages"]

    if soup:
        # Attempt to find the XML element by id, and convert it to details
        if "id" in ref_content:
            ref_tag = first(soup.select("ref#" + ref_content["id"]))
            if ref_tag:
                # Now remove tags that would be already part of the unknown reference by now
                for remove_tag in ["person-group", "year", "article-title",
                                   "elocation-id", "fpage", "lpage"]:
                    ref_tag = remove_tag_from_tag(ref_tag, remove_tag)
                # Add the remaining tag content comma separated
                for tag in first(raw_parser.element_citation(ref_tag)):
                    if node_text(tag) is not None:
                        if details != "":
                            details += ", "
                        details += node_text(tag)
    if details == "":
        return None
    else:
        return details