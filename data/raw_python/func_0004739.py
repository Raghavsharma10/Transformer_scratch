def render_twitter(text, **kwargs):
    """
    Strict template block for rendering twitter embeds.
    """
    author = render_author(**kwargs['author'])
    metadata = render_metadata(**kwargs['metadata'])
    image = render_image(**kwargs['image'])

    html = """
        <div class="attachment attachment-twitter">
            {author}
            <p class="twitter-content">{text}</p>
            {metadata}
            {image}
        </div>
    """.format(
        author=author,
        text=text,
        metadata=metadata,
        image=image
    ).strip()

    return html