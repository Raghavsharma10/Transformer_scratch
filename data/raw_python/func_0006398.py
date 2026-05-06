def get_charge(max_tdc, tdc_calibration_values, tdc_pixel_calibration):  # Return the charge from calibration
    ''' Interpolatet the TDC calibration for each pixel from 0 to max_tdc'''
    charge_calibration = np.zeros(shape=(80, 336, max_tdc))
    for column in range(80):
        for row in range(336):
            actual_pixel_calibration = tdc_pixel_calibration[column, row, :]
            if np.any(actual_pixel_calibration != 0) and np.any(np.isfinite(actual_pixel_calibration)):
                selected_measurements = np.isfinite(actual_pixel_calibration)  # Select valid calibration steps
                selected_actual_pixel_calibration = actual_pixel_calibration[selected_measurements]
                selected_tdc_calibration_values = tdc_calibration_values[selected_measurements]
                interpolation = interp1d(x=selected_actual_pixel_calibration, y=selected_tdc_calibration_values, kind='slinear', bounds_error=False, fill_value=0)
                charge_calibration[column, row, :] = interpolation(np.arange(max_tdc))
    return charge_calibration