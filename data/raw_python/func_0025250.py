def auto_migrate_storage_system(*, persistent_storage_system=None, new_persistent_storage_system=None, data_item_uuids=None, deletions: typing.List[uuid.UUID] = None, utilized_deletions: typing.Set[uuid.UUID] = None, ignore_older_files: bool = True):
    """Migrate items from the storage system to the object context.

    Files in data_item_uuids have already been loaded and are ignored (not migrated).

    Files in deletes have been deleted in object context and are ignored (not migrated) and then added
    to the utilized deletions list.

    Data items will have persistent_object_context set upon return, but caller will need to call finish_reading
    on each of the data items.
    """
    storage_handlers = persistent_storage_system.find_data_items()
    ReaderInfo = collections.namedtuple("ReaderInfo", ["properties", "changed_ref", "large_format", "storage_handler", "identifier"])
    reader_info_list = list()
    for storage_handler in storage_handlers:
        try:
            large_format = isinstance(storage_handler, HDF5Handler.HDF5Handler)
            properties = Migration.transform_to_latest(storage_handler.read_properties())
            reader_info = ReaderInfo(properties, [False], large_format, storage_handler, storage_handler.reference)
            reader_info_list.append(reader_info)
        except Exception as e:
            logging.debug("Error reading %s", storage_handler.reference)
            import traceback
            traceback.print_exc()
            traceback.print_stack()
    library_storage_properties = persistent_storage_system.library_storage_properties
    for deletion in copy.deepcopy(library_storage_properties.get("data_item_deletions", list())):
        if not deletion in deletions:
            deletions.append(deletion)
    preliminary_library_updates = dict()
    library_updates = dict()
    if not ignore_older_files:
        Migration.migrate_to_latest(reader_info_list, preliminary_library_updates)
    good_reader_info_list = list()
    count = len(reader_info_list)
    for index, reader_info in enumerate(reader_info_list):
        storage_handler = reader_info.storage_handler
        properties = reader_info.properties
        try:
            version = properties.get("version", 0)
            if version == DataItem.DataItem.writer_version:
                data_item_uuid = uuid.UUID(properties["uuid"])
                if not data_item_uuid in data_item_uuids:
                    if str(data_item_uuid) in deletions:
                        utilized_deletions.add(data_item_uuid)
                    else:
                        auto_migrate_data_item(reader_info, persistent_storage_system, new_persistent_storage_system, index, count)
                        good_reader_info_list.append(reader_info)
                        data_item_uuids.add(data_item_uuid)
                        library_update = preliminary_library_updates.get(data_item_uuid)
                        if library_update:
                            library_updates[data_item_uuid] = library_update
        except Exception as e:
            logging.debug("Error reading %s", storage_handler.reference)
            import traceback
            traceback.print_exc()
            traceback.print_stack()
    return good_reader_info_list, library_updates