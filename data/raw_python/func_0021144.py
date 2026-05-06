def display_date(d):
    """
    Render a date/datetime (d) as a date, using the SPECTATOR_DATE_FORMAT
    setting. Wrap the output in a <time> tag.

    Time tags: http://www.brucelawson.co.uk/2012/best-of-time/
    """
    stamp = d.strftime('%Y-%m-%d')
    visible_date = d.strftime(app_settings.DATE_FORMAT)

    return format_html('<time datetime="%(stamp)s">%(visible)s</time>' % {
                'stamp': stamp,
                'visible': visible_date
            })