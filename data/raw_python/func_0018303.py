def Image(props):
    """
    Inspired by:
    - https://github.com/torchbox/wagtail/blob/master/wagtail/wagtailimages/rich_text.py
    - https://github.com/torchbox/wagtail/blob/master/wagtail/wagtailimages/shortcuts.py
    - https://github.com/torchbox/wagtail/blob/master/wagtail/wagtailimages/formats.py
    """
    image_model = get_image_model()
    alignment = props.get('alignment', 'left')
    alt_text = props.get('altText', '')

    try:
        image = image_model.objects.get(id=props['id'])
    except image_model.DoesNotExist:
        return DOM.create_element('img', {'alt': alt_text})

    image_format = get_image_format(alignment)
    rendition = get_rendition_or_not_found(image, image_format.filter_spec)

    return DOM.create_element('img', dict(rendition.attrs_dict, **{
        'class': image_format.classnames,
        'src': rendition.url,
        'alt': alt_text,
    }))