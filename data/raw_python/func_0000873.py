def rewrite_subject_group(root, subjects, subject_group_type, overwrite=True):
    "add or rewrite subject tags inside subj-group tags"
    parent_tag_name = 'subj-group'
    tag_name = 'subject'
    wrap_tag_name = 'article-categories'
    tag_attribute = 'subj-group-type'
    # the parent tag where it should be found
    xpath_parent = './/front/article-meta/article-categories'
    # the wraping tag in case article-categories does not exist
    xpath_article_meta = './/front/article-meta'
    # the xpath to find the subject tags we are interested in
    xpath = './/{parent_tag_name}[@{tag_attribute}="{group_type}"]'.format(
        parent_tag_name=parent_tag_name,
        tag_attribute=tag_attribute,
        group_type=subject_group_type)

    count = 0
    # get the parent tag
    parent_tag = root.find(xpath_parent)
    if parent_tag is None:
        # parent tag not found, add one
        wrap_tag = root.find(xpath_article_meta)
        article_categories_tag = SubElement(wrap_tag, wrap_tag_name)
        parent_tag = article_categories_tag
    insert_index = 0
    # iterate all tags to find the index of the first tag we are interested in
    if parent_tag is not None:
        for tag_index, tag in enumerate(parent_tag.findall('*')):
            if tag.tag == parent_tag_name and tag.get(tag_attribute) == subject_group_type:
                insert_index = tag_index
                if overwrite is True:
                    # if overwriting use the first one found
                    break
            # if not overwriting, use the last one found + 1
            if overwrite is not True:
                insert_index += 1
    # remove the tag if overwriting the existing values
    if overwrite is True:
        # remove all the tags
        for tag in root.findall(xpath):
            parent_tag.remove(tag)
    # add the subjects
    for subject in subjects:
        subj_group_tag = Element(parent_tag_name)
        subj_group_tag.set(tag_attribute, subject_group_type)
        subject_tag = SubElement(subj_group_tag, tag_name)
        subject_tag.text = subject
        parent_tag.insert(insert_index, subj_group_tag)
        count += 1
        insert_index += 1
    return count