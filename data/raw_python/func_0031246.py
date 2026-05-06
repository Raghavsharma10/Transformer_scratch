def get_template_context(src, container="div", classes="", inner_classes="", alt="", background_image=False, no_css=False, aria_hidden=False):
    """Returns a template context for a flexible image template
    tag implementation."""
    context = {
        "container": container,
        "classes": classes,
        "aspect_padding_bottom": aspect_ratio_percent(src),
        "alt": alt,
        "background_image": background_image,
        "no_css": no_css,
        "inner_classes": inner_classes,
        "aria_hidden": aria_hidden,
    }

    # We can't do any of the srcset (or JS switching fallback) if we don't
    # have a thumbnail library installed.
    if not get_thumbnail_engine():
        context["image"] = src
        return context

    sizes = get_image_sizes(src)
    context["image_sizes"] = sizes

    # Set the first image in the list as the one to be rendered initially
    # (pre-JS-fallback). `if sizes` might not be a necessary check...
    context["image"] = sizes[0]

    context["image_sizes_json"] = json.dumps(sizes)
    srcset_items = ["{} {}w".format(size["url"], size["width"]) for size in sizes]

    context["image_sizes_srcset"] = ", ".join(srcset_items)
    return context