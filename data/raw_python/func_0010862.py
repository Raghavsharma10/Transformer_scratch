def add_header_info(data_api, struct_inflator):
    """ Add ancilliary header information to the structure.
	 :param data_api the interface to the decoded data
	 :param struct_inflator the interface to put the data into the client object
	 """
    struct_inflator.set_header_info(data_api.r_free,
                                    data_api.r_work,
                                    data_api.resolution,
                                    data_api.title,
                                    data_api.deposition_date,
                                    data_api.release_date,
                                    data_api.experimental_methods)