def add_routes(meteor_app, url_path='/importv2'):
    """
    Add two routes to the specified instance of :class:`meteorpi_server.MeteorApp` to implement the import API and allow
    for replication of data to this server.

    :param meteorpi_server.MeteorApp meteor_app:
        The :class:`meteorpi_server.MeteorApp` to which import routes should be added
    :param meteorpi_server.importer_api.BaseImportReceiver handler:
        A subclass of :class:`meteorpi_server.importer_api.BaseImportReceiver` which is used to handle the import. If
        not specified this defaults to an instance of :class:`meteorpi_server.importer_api.MeteorDatabaseImportReceiver`
        which will replicate any missing information from the import into the database attached to the meteor_app.
    :param string url_path:
        The base of the import routes for this application. Defaults to '/import' - routes will be created at this path
        and as import_path/data/<id> for binary data reception. Both paths only respond to POST requests and require
        that the requests are authenticated and that the authenticated user has the 'import' role.
    """
    app = meteor_app.app

    @app.route(url_path, methods=['POST'])
    @meteor_app.requires_auth(roles=['import'])
    def import_entities():
        """
        Receive an entity import request, using :class:`meteorpi_server.import_api.ImportRequest` to parse it, then
        passing the parsed request on to an instance of :class:`meteorpi_server.import_api.BaseImportReceiver` to deal
        with the possible import types.

        :return:
            A response, generally using one of the response_xxx methods in ImportRequest
        """
        db = meteor_app.get_db()
        handler = MeteorDatabaseImportReceiver(db=db)
        import_request = ImportRequest.process_request()
        if import_request.entity is None:
            return import_request.response_continue()
        if import_request.entity_type == 'file':
            response = handler.receive_file_record(import_request)
            handler.db.commit()
            db.close_db()
            if response is not None:
                return response
            else:
                return import_request.response_complete()
        elif import_request.entity_type == 'observation':
            response = handler.receive_observation(import_request)
            handler.db.commit()
            db.close_db()
            if response is not None:
                return response
            else:
                return import_request.response_complete()
        elif import_request.entity_type == 'metadata':
            response = handler.receive_metadata(import_request)
            handler.db.commit()
            db.close_db()
            if response is not None:
                return response
            else:
                return import_request.response_continue()
        else:
            db.close_db()
            return import_request.response_failed("Unknown import request")

    @app.route('{0}/data/<file_id_hex>/<md5_hex>'.format(url_path), methods=['POST'])
    @meteor_app.requires_auth(roles=['import'])
    def import_file_data(file_id_hex, md5_hex):
        """
        Receive a file upload, passing it to the handler if it contains the appropriate information

        :param string file_id_hex:
            The hex representation of the :class:`meteorpi_model.FileRecord` to which this data belongs.
        """
        db = meteor_app.get_db()
        handler = MeteorDatabaseImportReceiver(db=db)
        file_id = file_id_hex
        file_data = request.files['file']
        if file_data:
            handler.receive_file_data(file_id=file_id, file_data=file_data, md5_hex=md5_hex)
        db.close_db()
        return ImportRequest.response_continue_after_file()