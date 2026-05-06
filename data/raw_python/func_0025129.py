def interpolate_mrms_day(start_date, variable, interp_type, mrms_path, map_filename, out_path):
    """
    For a given day, this module interpolates hourly MRMS data to a specified latitude and 
    longitude grid, and saves the interpolated grids to CF-compliant netCDF4 files.
    
    Args:
        start_date (datetime.datetime): Date of data being interpolated
        variable (str): MRMS variable
        interp_type (str): Whether to use maximum neighbor or spline
        mrms_path (str): Path to top-level directory of MRMS GRIB2 files
        map_filename (str): Name of the map filename. Supports ARPS map file format and netCDF files containing latitude
            and longitude variables
        out_path (str): Path to location where interpolated netCDF4 files are saved.
    """
    try:
        print(start_date, variable)
        end_date = start_date + timedelta(hours=23)
        mrms = MRMSGrid(start_date, end_date, variable, mrms_path)
        if mrms.data is not None:
            if map_filename[-3:] == "map":
                mapping_data = make_proj_grids(*read_arps_map_file(map_filename))
                mrms.interpolate_to_netcdf(mapping_data['lon'], mapping_data['lat'], out_path, interp_type=interp_type)
            elif map_filename[-3:] == "txt":
                mapping_data = make_proj_grids(*read_ncar_map_file(map_filename))
                mrms.interpolate_to_netcdf(mapping_data["lon"], mapping_data["lat"], out_path, interp_type=interp_type)
            else:
                lon, lat = load_map_coordinates(map_filename)
                mrms.interpolate_to_netcdf(lon, lat, out_path, interp_type=interp_type)
    except Exception as e:
        # This exception catches any errors when run in multiprocessing, prints the stack trace,
        # and ends the process. Otherwise the process will stall.
        print(traceback.format_exc())
        raise e