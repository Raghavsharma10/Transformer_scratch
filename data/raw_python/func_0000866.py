def supp_asset(tag):
    """
    Given a supplementary-material tag, the asset value depends on
    its label text. This also informs in what order (its ordinal) it
    has depending on how many of each type is present
    """
    # Default
    asset = 'supp'
    if first(extract_nodes(tag, "label")):
        label_text = node_text(first(extract_nodes(tag, "label"))).lower()
        # Keyword match the label
        if label_text.find('code') > 0:
            asset = 'code'
        elif label_text.find('data') > 0:
            asset = 'data'
    return asset