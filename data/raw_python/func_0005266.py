def search_tags_as_filters(tags):
    """Get different tags as dicts ready to use as dropdown lists."""
    # set dicts
    actions = {}
    contacts = {}
    formats = {}
    inspire = {}
    keywords = {}
    licenses = {}
    md_types = dict()
    owners = defaultdict(str)
    srs = {}
    unused = {}
    # 0/1 values
    compliance = 0
    type_dataset = 0
    # parsing tags
    print(len(tags.keys()))
    i = 0
    for tag in sorted(tags.keys()):
        i += 1
        # actions
        if tag.startswith("action"):
            actions[tags.get(tag, tag)] = tag
            continue
        # compliance INSPIRE
        elif tag.startswith("conformity"):
            compliance = 1
            continue
        # contacts
        elif tag.startswith("contact"):
            contacts[tags.get(tag)] = tag
            continue
        # formats
        elif tag.startswith("format"):
            formats[tags.get(tag)] = tag
            continue
        # INSPIRE themes
        elif tag.startswith("keyword:inspire"):
            inspire[tags.get(tag)] = tag
            continue
        # keywords
        elif tag.startswith("keyword:isogeo"):
            keywords[tags.get(tag)] = tag
            continue
        # licenses
        elif tag.startswith("license"):
            licenses[tags.get(tag)] = tag
            continue
        # owners
        elif tag.startswith("owner"):
            owners[tags.get(tag)] = tag
            continue
        # SRS
        elif tag.startswith("coordinate-system"):
            srs[tags.get(tag)] = tag
            continue
        # types
        elif tag.startswith("type"):
            md_types[tags.get(tag)] = tag
            if tag in ("type:vector-dataset", "type:raster-dataset"):
                type_dataset += 1
            else:
                pass
            continue
        # ignored tags
        else:
            unused[tags.get(tag)] = tag
            continue

    # override API tags to allow all datasets filter - see #
    if type_dataset == 2:
        md_types["Donnée"] = "type:dataset"
    else:
        pass
    # printing
    # print("There are:"
    #       "\n{} actions"
    #       "\n{} contacts"
    #       "\n{} formats"
    #       "\n{} INSPIRE themes"
    #       "\n{} keywords"
    #       "\n{} licenses"
    #       "\n{} owners"
    #       "\n{} SRS"
    #       "\n{} types"
    #       "\n{} unused".format(len(actions),
    #                            len(contacts),
    #                            len(formats),
    #                            len(inspire),
    #                            len(keywords),
    #                            len(licenses),
    #                            len(owners),
    #                            len(srs),
    #                            len(md_types),
    #                            len(unused)
    #                            ))
    # storing dicts
    tags_parsed = {
        "actions": actions,
        "compliance": compliance,
        "contacts": contacts,
        "formats": formats,
        "inspire": inspire,
        "keywords": keywords,
        "licenses": licenses,
        "owners": owners,
        "srs": srs,
        "types": md_types,
        "unused": unused,
    }

    # method ending
    return tags_parsed