def tag_media_sibling_ordinal(tag):
    """
    Count sibling ordinal differently depending on if the
    mimetype is video or not
    """
    if hasattr(tag, 'name') and tag.name != 'media':
        return None

    nodenames = ['fig','supplementary-material','sub-article']
    first_parent_tag = first_parent(tag, nodenames)

    sibling_ordinal = None

    if first_parent_tag:
        # Start counting at 0
        sibling_ordinal = 0
        for media_tag in first_parent_tag.find_all(tag.name):
            if 'mimetype' in tag.attrs and tag['mimetype'] == 'video':
                # Count all video type media tags
                if 'mimetype' in media_tag.attrs and tag['mimetype'] == 'video':
                    sibling_ordinal += 1
                if media_tag == tag:
                    break

            else:
                # Count all non-video type media tags
                if (('mimetype' not in media_tag.attrs)
                    or ('mimetype' in media_tag.attrs and tag['mimetype'] != 'video')):
                    sibling_ordinal += 1
                if media_tag == tag:
                    break
    else:
        # Start counting at 1
        sibling_ordinal = 1
        for prev_tag in tag.find_all_previous(tag.name):
            if not first_parent(prev_tag, nodenames):
                if 'mimetype' in tag.attrs and tag['mimetype'] == 'video':
                    # Count all video type media tags
                    if supp_asset(prev_tag) == supp_asset(tag) and 'mimetype' in prev_tag.attrs:
                        sibling_ordinal += 1
                else:
                    if supp_asset(prev_tag) == supp_asset(tag) and 'mimetype' not in prev_tag.attrs:
                        sibling_ordinal += 1

    return sibling_ordinal