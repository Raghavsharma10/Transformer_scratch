def _applyTargetState(targetState, md, httpclient):
    """
    compares the current device state against the targetStateProvider and issues updates as necessary to ensure the
    device is
    at that state.
    :param md:
    :param targetState: the target state.
    :param httpclient: the http client
    :return:
    """
    anyUpdate = False
    if md['fs'] != targetState.fs:
        logger.info("Updating fs from " + str(md['fs']) + " to " + str(targetState.fs) + " for " + md['name'])
        anyUpdate = True

    if md['samplesPerBatch'] != targetState.samplesPerBatch:
        logger.info("Updating samplesPerBatch from " + str(md['samplesPerBatch']) + " to " + str(
            targetState.samplesPerBatch) + " for " + md['name'])
        anyUpdate = True

    if md['gyroEnabled'] != targetState.gyroEnabled:
        logger.info("Updating gyroEnabled from " + str(md['gyroEnabled']) + " to " + str(
            targetState.gyroEnabled) + " for " + md['name'])
        anyUpdate = True

    if md['gyroSens'] != targetState.gyroSens:
        logger.info(
            "Updating gyroSens from " + str(md['gyroSens']) + " to " + str(targetState.gyroSens) + " for " + md[
                'name'])
        anyUpdate = True

    if md['accelerometerEnabled'] != targetState.accelerometerEnabled:
        logger.info("Updating accelerometerEnabled from " + str(md['accelerometerEnabled']) + " to " + str(
            targetState.accelerometerEnabled) + " for " + md['name'])
        anyUpdate = True

    if md['accelerometerSens'] != targetState.accelerometerSens:
        logger.info("Updating accelerometerSens from " + str(md['accelerometerSens']) + " to " + str(
            targetState.accelerometerSens) + " for " + md['name'])
        anyUpdate = True

    if anyUpdate:
        payload = marshal(targetState, targetStateFields)
        logger.info("Applying target state change " + md['name'] + " - " + str(payload))
        if RecordingDeviceStatus.INITIALISED.name == md.get('status'):
            try:
                httpclient.patch(md['serviceURL'], json=payload)
            except Exception as e:
                logger.exception(e)
        else:
            logger.warning("Ignoring target state change until " + md['name'] + " is idle, currently " + md['status'])
    else:
        logger.debug("Device " + md['name'] + " is at target state, we continue")