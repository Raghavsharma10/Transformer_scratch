def convert_data_element_to_data_and_metadata_1(data_element) -> DataAndMetadata.DataAndMetadata:
    """Convert a data element to xdata. No data copying occurs.

    The data element can have the following keys:
        data (required)
        is_sequence, collection_dimension_count, datum_dimension_count (optional description of the data)
        spatial_calibrations (optional list of spatial calibration dicts, scale, offset, units)
        intensity_calibration (optional intensity calibration dict, scale, offset, units)
        metadata (optional)
        properties (get stored into metadata.hardware_source)
        one of either timestamp or datetime_modified
        if datetime_modified (dst, tz) it is converted and used as timestamp
            then timezone gets stored into metadata.description.timezone.
    """
    # data. takes ownership.
    data = data_element["data"]
    dimensional_shape = Image.dimensional_shape_from_data(data)
    is_sequence = data_element.get("is_sequence", False)
    dimension_count = len(Image.dimensional_shape_from_data(data))
    adjusted_dimension_count = dimension_count - (1 if is_sequence else 0)
    collection_dimension_count = data_element.get("collection_dimension_count", 2 if adjusted_dimension_count in (3, 4) else 0)
    datum_dimension_count = data_element.get("datum_dimension_count", adjusted_dimension_count - collection_dimension_count)
    data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

    # dimensional calibrations
    dimensional_calibrations = None
    if "spatial_calibrations" in data_element:
        dimensional_calibrations_list = data_element.get("spatial_calibrations")
        if len(dimensional_calibrations_list) == len(dimensional_shape):
            dimensional_calibrations = list()
            for dimension_calibration in dimensional_calibrations_list:
                offset = float(dimension_calibration.get("offset", 0.0))
                scale = float(dimension_calibration.get("scale", 1.0))
                units = dimension_calibration.get("units", "")
                units = str(units) if units is not None else str()
                if scale != 0.0:
                    dimensional_calibrations.append(Calibration.Calibration(offset, scale, units))
                else:
                    dimensional_calibrations.append(Calibration.Calibration())

    # intensity calibration
    intensity_calibration = None
    if "intensity_calibration" in data_element:
        intensity_calibration_dict = data_element.get("intensity_calibration")
        offset = float(intensity_calibration_dict.get("offset", 0.0))
        scale = float(intensity_calibration_dict.get("scale", 1.0))
        units = intensity_calibration_dict.get("units", "")
        units = str(units) if units is not None else str()
        if scale != 0.0:
            intensity_calibration = Calibration.Calibration(offset, scale, units)

    # properties (general tags)
    metadata = dict()
    if "metadata" in data_element:
        metadata.update(Utility.clean_dict(data_element.get("metadata")))
    if "properties" in data_element and data_element["properties"]:
        hardware_source_metadata = metadata.setdefault("hardware_source", dict())
        hardware_source_metadata.update(Utility.clean_dict(data_element.get("properties")))

    # dates are _local_ time and must use this specific ISO 8601 format. 2013-11-17T08:43:21.389391
    # time zones are offsets (east of UTC) in the following format "+HHMM" or "-HHMM"
    # daylight savings times are time offset (east of UTC) in format "+MM" or "-MM"
    # timezone is for conversion and is the Olson timezone string.
    # datetime.datetime.strptime(datetime.datetime.isoformat(datetime.datetime.now()), "%Y-%m-%dT%H:%M:%S.%f" )
    # datetime_modified, datetime_modified_tz, datetime_modified_dst, datetime_modified_tzname is the time at which this image was modified.
    # datetime_original, datetime_original_tz, datetime_original_dst, datetime_original_tzname is the time at which this image was created.
    timestamp = data_element.get("timestamp", datetime.datetime.utcnow())
    datetime_item = data_element.get("datetime_modified", Utility.get_datetime_item_from_utc_datetime(timestamp))

    local_datetime = Utility.get_datetime_from_datetime_item(datetime_item)
    dst_value = datetime_item.get("dst", "+00")
    tz_value = datetime_item.get("tz", "+0000")
    timezone = datetime_item.get("timezone")
    time_zone = { "dst": dst_value, "tz": tz_value}
    if timezone is not None:
        time_zone["timezone"] = timezone
    # note: dst is informational only; tz already include dst
    tz_adjust = (int(tz_value[1:3]) * 60 + int(tz_value[3:5])) * (-1 if tz_value[0] == '-' else 1)
    utc_datetime = local_datetime - datetime.timedelta(minutes=tz_adjust)  # tz_adjust already contains dst_adjust
    timestamp = utc_datetime

    return DataAndMetadata.new_data_and_metadata(data,
                                                 intensity_calibration=intensity_calibration,
                                                 dimensional_calibrations=dimensional_calibrations,
                                                 metadata=metadata,
                                                 timestamp=timestamp,
                                                 data_descriptor=data_descriptor,
                                                 timezone=timezone,
                                                 timezone_offset=tz_value)