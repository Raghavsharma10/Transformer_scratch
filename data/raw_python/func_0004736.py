def render_author(**kwargs):
    """
    Unstrict template block for rendering authors:
    <div class="author">
        <img class="author-avatar" src="{author_avatar}">
        <p class="author-name">
            <a href="{author_link}">{author_name}</a>
        </p>
        <p class="user-handle">{author_handle}</p>
    </div>
    """
    html = '<div class="user">'

    author_avatar = kwargs.get('author_avatar', None)
    if author_avatar:
        html += '<img class="user-avatar" src="{}">'.format(author_avatar)

    author_name = kwargs.get('author_name', None)
    if author_name:
        html += '<p class="user-name">'

        author_link = kwargs.get('author_link', None)
        if author_link:
            html += '<a href="{author_link}">{author_name}</a>'.format(
                author_link=author_link,
                author_name=author_name
            )
        else:
            html += author_name

        html += '</p>'

    author_handle = kwargs.get('author_handle', None)
    if author_handle:
        html += '<p class="user-handle">{}</p>'.format(author_handle)

    html += '</div>'