def references_json_authors(ref_authors, ref_content):
    "build the authors for references json here for testability"
    all_authors = references_authors(ref_authors)
    if all_authors != {}:
        if ref_content.get("type") in ["conference-proceeding", "journal", "other",
                                           "periodical", "preprint", "report", "web"]:
            for author_type in ["authors", "authorsEtAl"]:
                set_if_value(ref_content, author_type, all_authors.get(author_type))
        elif ref_content.get("type") in ["book", "book-chapter"]:
            for author_type in ["authors", "authorsEtAl", "editors", "editorsEtAl"]:
                set_if_value(ref_content, author_type, all_authors.get(author_type))
        elif ref_content.get("type") in ["clinical-trial"]:
            # Always set as authors, once,  then add the authorsType
            for author_type in ["authors", "collaborators", "sponsors"]:
                if "authorsType" not in ref_content and all_authors.get(author_type):
                    set_if_value(ref_content, "authors", all_authors.get(author_type))
                    set_if_value(ref_content, "authorsEtAl", all_authors.get(author_type + "EtAl"))
                    ref_content["authorsType"] = author_type
        elif ref_content.get("type") in ["data", "software"]:
            for author_type in ["authors", "authorsEtAl",
                                "compilers", "compilersEtAl", "curators", "curatorsEtAl"]:
                set_if_value(ref_content, author_type, all_authors.get(author_type))
        elif ref_content.get("type") in ["patent"]:
            for author_type in ["inventors", "inventorsEtAl", "assignees", "assigneesEtAl"]:
                set_if_value(ref_content, author_type, all_authors.get(author_type))
        elif ref_content.get("type") in ["thesis"]:
            # Convert list to a non-list
            if all_authors.get("authors") and len(all_authors.get("authors")) > 0:
                ref_content["author"] = all_authors.get("authors")[0]
    return ref_content