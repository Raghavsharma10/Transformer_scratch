def render_metadata(**kwargs):
    """
    Unstrict template block for rendering metadata:
    <div class="metadata">
        <img class="metadata-logo" src="{service_logo}">
        <p class="metadata-name">{service_name}</p>
        <p class="metadata-timestamp">
            <a href="{timestamp_link}">{timestamp}</a>
        </p>
    </div>
    """
    html = '<div class="metadata">'

    service_logo = kwargs.get('service_logo', None)
    if service_logo:
        html += '<img class="metadata-logo" src="{}">'.format(service_logo)

    service_name = kwargs.get('service_name', None)
    if service_name:
        html += '<p class="metadata-name">{}</p>'.format(service_name)

    timestamp = kwargs.get('timestamp', None)
    if timestamp:
        html += '<p class="user-name">'

        timestamp_link = kwargs.get('timestamp_link', None)
        if timestamp_link:
            html += '<a href="{timestamp_link}">{timestamp}</a>'.format(
                timestamp_link=timestamp_link,
                timestamp=timestamp
            )
        else:
            html += timestamp

        html += '</p>'

    html += '</div>'