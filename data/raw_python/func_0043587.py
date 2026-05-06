def example(index):
    """Index page."""
    pid = PersistentIdentifier.query.filter_by(id=index).one()
    record = RecordMetadata.query.filter_by(id=pid.object_uuid).first()

    return render_template("app/detail.html", record=record.json, pid=pid,
                           title="Demosite Invenio Org")