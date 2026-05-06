def GetPupil(self):
    """Retrieve pupil data
    """
    pupil_data = _co.namedtuple('pupil_data', ['ZemaxApertureType',
                                               'ApertureValue',
                                               'entrancePupilDiameter',
                                               'entrancePupilPosition',
                                               'exitPupilDiameter',
                                               'exitPupilPosition',
                                               'ApodizationType',
                                               'ApodizationFactor'])
    data = self._ilensdataeditor.GetPupil()
    return pupil_data(*data)