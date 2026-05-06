def _normalize_license(data):
    """try to get SDPX license"""
    license = data.get('license', None)
    if not license:
        # try to get license from classifiers
        license = _license_from_classifiers(data)
    if license:
        if license in SDPX_LICENSES.keys():
            data['license'] = SDPX_LICENSES[license]
        else:
            data['license'] = "%s (FIXME:No SPDX)" % (license)
    else:
        data['license'] = ""