def write(path, doc, mode=MODE_TSV, **kwargs):
    ''' Helper function to write doc to TTL-TXT format '''
    if mode == MODE_TSV:
        with TxtWriter.from_path(path) as writer:
            writer.write_doc(doc)
    elif mode == MODE_JSON:
        write_json(path, doc, **kwargs)