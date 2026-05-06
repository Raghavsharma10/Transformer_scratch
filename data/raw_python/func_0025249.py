def read_library(persistent_storage_system, ignore_older_files) -> typing.Dict:
    """Read data items from the data reference handler and return as a list.

    Data items will have persistent_object_context set upon return, but caller will need to call finish_reading
    on each of the data items.
    """
    data_item_uuids = set()
    utilized_deletions = set()  # the uuid's skipped due to being deleted
    deletions = list()

    reader_info_list, library_updates = auto_migrate_storage_system(persistent_storage_system=persistent_storage_system,
                                                                    new_persistent_storage_system=persistent_storage_system,
                                                                    data_item_uuids=data_item_uuids,
                                                                    deletions=deletions,
                                                                    utilized_deletions=utilized_deletions,
                                                                    ignore_older_files=ignore_older_files)

    # next, for each auto migration, create a temporary storage system and read items from that storage system
    # using auto_migrate_storage_system. the data items returned will have been copied to the current storage
    # system (persistent object context).
    for auto_migration in reversed(persistent_storage_system.get_auto_migrations()):
        old_persistent_storage_system = FileStorageSystem(auto_migration.library_path, auto_migration.paths) if auto_migration.paths else auto_migration.storage_system
        new_reader_info_list, new_library_updates = auto_migrate_storage_system(persistent_storage_system=old_persistent_storage_system,
                                                                                new_persistent_storage_system=persistent_storage_system,
                                                                                data_item_uuids=data_item_uuids,
                                                                                deletions=deletions,
                                                                                utilized_deletions=utilized_deletions,
                                                                                ignore_older_files=ignore_older_files)
        reader_info_list.extend(new_reader_info_list)
        library_updates.update(new_library_updates)

    assert len(reader_info_list) == len(data_item_uuids)

    library_storage_properties = persistent_storage_system.library_storage_properties

    for reader_info in reader_info_list:
        properties = reader_info.properties
        properties = Utility.clean_dict(copy.deepcopy(properties) if properties else dict())
        version = properties.get("version", 0)
        if version == DataItem.DataItem.writer_version:
            data_item_uuid = uuid.UUID(properties.get("uuid", uuid.uuid4()))
            library_update = library_updates.get(data_item_uuid, dict())
            library_storage_properties.setdefault("connections", list()).extend(library_update.get("connections", list()))
            library_storage_properties.setdefault("computations", list()).extend(library_update.get("computations", list()))
            library_storage_properties.setdefault("display_items", list()).extend(library_update.get("display_items", list()))

    # mark deletions that need to be tracked because they've been deleted but are also present in older libraries
    # and would be migrated during reading unless they explicitly are prevented from doing so (via data_item_deletions).
    # utilized deletions are the ones that were attempted; if nothing was attempted, then no reason to track it anymore
    # since there is nothing to migrate in the future.
    library_storage_properties["data_item_deletions"] = [str(uuid_) for uuid_ in utilized_deletions]

    connections_list = library_storage_properties.get("connections", list())
    assert len(connections_list) == len({connection.get("uuid") for connection in connections_list})

    computations_list = library_storage_properties.get("computations", list())
    assert len(computations_list) == len({computation.get("uuid") for computation in computations_list})

    # migrations

    if library_storage_properties.get("version", 0) < 2:
        for data_group_properties in library_storage_properties.get("data_groups", list()):
            data_group_properties.pop("data_groups")
            display_item_references = data_group_properties.setdefault("display_item_references", list())
            data_item_uuid_strs = data_group_properties.pop("data_item_uuids", list())
            for data_item_uuid_str in data_item_uuid_strs:
                for display_item_properties in library_storage_properties.get("display_items", list()):
                    data_item_references = [d.get("data_item_reference", None) for d in display_item_properties.get("display_data_channels", list())]
                    if data_item_uuid_str in data_item_references:
                        display_item_references.append(display_item_properties["uuid"])
        data_item_uuid_to_display_item_uuid_map = dict()
        data_item_uuid_to_display_item_dict_map = dict()
        display_to_display_item_map = dict()
        display_to_display_data_channel_map = dict()
        for display_item_properties in library_storage_properties.get("display_items", list()):
            display_to_display_item_map[display_item_properties["display"]["uuid"]] = display_item_properties["uuid"]
            display_to_display_data_channel_map[display_item_properties["display"]["uuid"]] = display_item_properties["display_data_channels"][0]["uuid"]
            data_item_references = [d.get("data_item_reference", None) for d in display_item_properties.get("display_data_channels", list())]
            for data_item_uuid_str in data_item_references:
                data_item_uuid_to_display_item_uuid_map.setdefault(data_item_uuid_str, display_item_properties["uuid"])
                data_item_uuid_to_display_item_dict_map.setdefault(data_item_uuid_str, display_item_properties)
            display_item_properties.pop("display", None)
        for workspace_properties in library_storage_properties.get("workspaces", list()):
            def replace1(d):
                if "children" in d:
                    for dd in d["children"]:
                        replace1(dd)
                if "data_item_uuid" in d:
                    data_item_uuid_str = d.pop("data_item_uuid")
                    display_item_uuid_str = data_item_uuid_to_display_item_uuid_map.get(data_item_uuid_str)
                    if display_item_uuid_str:
                        d["display_item_uuid"] = display_item_uuid_str
            replace1(workspace_properties["layout"])
        for connection_dict in library_storage_properties.get("connections", list()):
            source_uuid_str = connection_dict["source_uuid"]
            if connection_dict["type"] == "interval-list-connection":
                connection_dict["source_uuid"] = display_to_display_item_map.get(source_uuid_str, None)
            if connection_dict["type"] == "property-connection" and connection_dict["source_property"] == "slice_interval":
                connection_dict["source_uuid"] = display_to_display_data_channel_map.get(source_uuid_str, None)

        def fix_specifier(specifier_dict):
            if specifier_dict.get("type") in ("data_item", "display_xdata", "cropped_xdata", "cropped_display_xdata", "filter_xdata", "filtered_xdata"):
                if specifier_dict.get("uuid") in data_item_uuid_to_display_item_dict_map:
                    specifier_dict["uuid"] = data_item_uuid_to_display_item_dict_map[specifier_dict["uuid"]]["display_data_channels"][0]["uuid"]
                else:
                    specifier_dict.pop("uuid", None)
            if specifier_dict.get("type") == "data_item":
                specifier_dict["type"] = "data_source"
            if specifier_dict.get("type") == "data_item_object":
                specifier_dict["type"] = "data_item"
            if specifier_dict.get("type") == "region":
                specifier_dict["type"] = "graphic"

        for computation_dict in library_storage_properties.get("computations", list()):
            for variable_dict in computation_dict.get("variables", list()):
                if "specifier" in variable_dict:
                    specifier_dict = variable_dict["specifier"]
                    if specifier_dict is not None:
                        fix_specifier(specifier_dict)
                if "secondary_specifier" in variable_dict:
                    specifier_dict = variable_dict["secondary_specifier"]
                    if specifier_dict is not None:
                        fix_specifier(specifier_dict)
            for result_dict in computation_dict.get("results", list()):
                fix_specifier(result_dict["specifier"])

        library_storage_properties["version"] = DocumentModel.DocumentModel.library_version

    # TODO: add consistency checks: no duplicated items [by uuid] such as connections or computations or data items

    assert library_storage_properties["version"] == DocumentModel.DocumentModel.library_version

    persistent_storage_system.rewrite_properties(library_storage_properties)

    properties = copy.deepcopy(library_storage_properties)

    for reader_info in reader_info_list:
        data_item_properties = Utility.clean_dict(reader_info.properties if reader_info.properties else dict())
        if data_item_properties.get("version", 0) == DataItem.DataItem.writer_version:
            data_item_properties["__large_format"] = reader_info.large_format
            data_item_properties["__identifier"] = reader_info.identifier
            properties.setdefault("data_items", list()).append(data_item_properties)

    def data_item_created(data_item_properties: typing.Mapping) -> str:
        return data_item_properties.get("created", "1900-01-01T00:00:00.000000")

    properties["data_items"] = sorted(properties.get("data_items", list()), key=data_item_created)

    return properties