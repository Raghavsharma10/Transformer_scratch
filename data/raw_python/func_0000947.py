def body_block_image_content(tag):
    "format a graphic or inline-graphic into a body block json format"
    image_content = OrderedDict()
    if tag:
        copy_attribute(tag.attrs, 'xlink:href', image_content, 'uri')
        if "uri" in image_content:
            # todo!! alt
            set_if_value(image_content, "alt", "")
    return image_content