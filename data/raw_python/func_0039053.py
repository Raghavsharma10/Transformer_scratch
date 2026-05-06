def set_org_processor(request):
    """
    Simple context processor that automatically sets 'org' on the context if it
    is present in the request.
    """
    if getattr(request, "org", None):
        org = request.org
        pattern_bg = org.backgrounds.filter(is_active=True, background_type="P")
        pattern_bg = pattern_bg.order_by("-pk").first()
        banner_bg = org.backgrounds.filter(is_active=True, background_type="B")
        banner_bg = banner_bg.order_by("-pk").first()

        return dict(org=org, pattern_bg=pattern_bg, banner_bg=banner_bg)
    else:
        return dict()