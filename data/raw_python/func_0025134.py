def get_data_generator_by_id(hardware_source_id, sync=True):
    """
        Return a generator for data.

        :param bool sync: whether to wait for current frame to finish then collect next frame

        NOTE: a new ndarray is created for each call.
    """
    hardware_source = HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
    def get_last_data():
        return hardware_source.get_next_xdatas_to_finish()[0].data.copy()
    yield get_last_data