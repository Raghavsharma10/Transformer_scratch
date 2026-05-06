def author_affiliations(author, html_flag=True):
    """compile author affiliations for json output"""

    # Configure the XML to HTML conversion preference for shorthand use below
    convert = lambda xml_string: xml_to_html(html_flag, xml_string)

    affilations = []

    if author.get("affiliations"):
        for affiliation in author.get("affiliations"):
            affiliation_json = OrderedDict()
            affiliation_json["name"] = []
            if affiliation.get("dept"):
                affiliation_json["name"].append(convert(affiliation.get("dept")))
            if affiliation.get("institution") and affiliation.get("institution").strip() != '':
                affiliation_json["name"].append(convert(affiliation.get("institution")))
            # Remove if empty
            if affiliation_json["name"] == []:
                del affiliation_json["name"]

            if ((affiliation.get("city") and affiliation.get("city").strip() != '')
                or affiliation.get("country") and affiliation.get("country").strip() != ''):
                affiliation_address = OrderedDict()
                affiliation_address["formatted"] = []
                affiliation_address["components"] = OrderedDict()
                if affiliation.get("city") and affiliation.get("city").strip() != '':
                    affiliation_address["formatted"].append(affiliation.get("city"))
                    affiliation_address["components"]["locality"] = []
                    affiliation_address["components"]["locality"].append(affiliation.get("city"))
                if affiliation.get("country") and affiliation.get("country").strip() != '':
                    affiliation_address["formatted"].append(affiliation.get("country"))
                    affiliation_address["components"]["country"] = affiliation.get("country")
                # Add if not empty
                if affiliation_address != {}:
                    affiliation_json["address"] = affiliation_address

            # Add if not empty
            if affiliation_json != {}:
                affilations.append(affiliation_json)

    if affilations != []:
        return affilations
    else:
        return None