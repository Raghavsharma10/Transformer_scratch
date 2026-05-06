def get_charge_calibration(calibation_file, max_tdc):
    ''' Open the hit or calibration file and return the calibration per pixel'''
    with tb.open_file(calibation_file, mode="r") as in_file_calibration_h5:
        tdc_calibration = in_file_calibration_h5.root.HitOrCalibration[:, :, :, 1]
        tdc_calibration_values = in_file_calibration_h5.root.HitOrCalibration.attrs.scan_parameter_values[:]
    return get_charge(max_tdc, tdc_calibration_values, tdc_calibration)