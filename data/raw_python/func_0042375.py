def collection(name=None):
    """Render the collection page.

    It renders it either with a collection specific template (aka
    collection_{collection_name}.html) or with the default collection
    template (collection.html).
    """
    if name is None:
        collection = Collection.query.get_or_404(1)
    else:
        collection = Collection.query.filter(
            Collection.name == name).first_or_404()

    # TODO add breadcrumbs
    # breadcrumbs = current_breadcrumbs + collection.breadcrumbs(ln=g.ln)[1:]
    return render_template([
        'invenio_collections/collection_{0}.html'.format(collection.id),
        'invenio_collections/collection_{0}.html'.format(slugify(name, '_')),
        current_app.config['COLLECTIONS_DEFAULT_TEMPLATE']
    ], collection=collection)