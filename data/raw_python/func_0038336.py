def _autofill_spec_record(record):
    """
    Returns an astropy table with columns auto-filled from FITS header

    Parameters
    ----------
    record: astropy.io.fits.table.table.Row
        The spectrum table row to scrape

    Returns
    -------
    record: astropy.io.fits.table.table.Row
        The spectrum table row with possible new rows inserted
    """
    try:
        record['filename'] = os.path.basename(record['spectrum'])

        if record['spectrum'].endswith('.fits'):
            header = pf.getheader(record['spectrum'])

            # Wavelength units
            if not record['wavelength_units']:
                try:
                    record['wavelength_units'] = header['XUNITS']
                except KeyError:
                    try:
                        if header['BUNIT']: record['wavelength_units'] = 'um'
                    except KeyError:
                        pass
            if 'microns' in record['wavelength_units'] or 'Microns' in record['wavelength_units'] or 'um' in record[
                'wavelength_units']:
                record['wavelength_units'] = 'um'

            # Flux units
            if not record['flux_units']:
                try:
                    record['flux_units'] = header['YUNITS'].replace(' ', '')
                except KeyError:
                    try:
                        record['flux_units'] = header['BUNIT'].replace(' ', '')
                    except KeyError:
                        pass
            if 'erg' in record['flux_units'] and 'A' in record['flux_units']:
                record['flux_units'] = 'ergs-1cm-2A-1' if 'erg' in record['flux_units'] and 'A' in record['flux_units'] \
                    else 'ergs-1cm-2um-1' if 'erg' in record['flux_units'] and 'um' in record['flux_units'] \
                    else 'Wm-2um-1' if 'W' in record['flux_units'] and 'um' in record['flux_units'] \
                    else 'Wm-2A-1' if 'W' in record['flux_units'] and 'A' in record['flux_units'] \
                    else ''

            # Observation date
            if not record['obs_date']:
                try:
                    record['obs_date'] = header['DATE_OBS']
                except KeyError:
                    try:
                        record['obs_date'] = header['DATE-OBS']
                    except KeyError:
                        try:
                            record['obs_date'] = header['DATE']
                        except KeyError:
                            pass

            # Telescope id
            if not record['telescope_id']:
                try:
                    n = header['TELESCOP'].lower() if isinstance(header['TELESCOP'], str) else ''
                    record['telescope_id'] = 5 if 'hst' in n \
                        else 6 if 'spitzer' in n \
                        else 7 if 'irtf' in n \
                        else 9 if 'keck' in n and 'ii' in n \
                        else 8 if 'keck' in n and 'i' in n \
                        else 10 if 'kp' in n and '4' in n \
                        else 11 if 'kp' in n and '2' in n \
                        else 12 if 'bok' in n \
                        else 13 if 'mmt' in n \
                        else 14 if 'ctio' in n and '1' in n \
                        else 15 if 'ctio' in n and '4' in n \
                        else 16 if 'gemini' in n and 'north' in n \
                        else 17 if 'gemini' in n and 'south' in n \
                        else 18 if ('vlt' in n and 'U2' in n) \
                        else 19 if '3.5m' in n \
                        else 20 if 'subaru' in n \
                        else 21 if ('mag' in n and 'ii' in n) or ('clay' in n) \
                        else 22 if ('mag' in n and 'i' in n) or ('baade' in n) \
                        else 23 if ('eso' in n and '1m' in n) \
                        else 24 if 'cfht' in n \
                        else 25 if 'ntt' in n \
                        else 26 if ('palomar' in n and '200-inch' in n) \
                        else 27 if 'pan-starrs' in n \
                        else 28 if ('palomar' in n and '60-inch' in n) \
                        else 29 if ('ctio' in n and '0.9m' in n) \
                        else 30 if 'soar' in n \
                        else 31 if ('vlt' in n and 'U3' in n) \
                        else 32 if ('vlt' in n and 'U4' in n) \
                        else 33 if 'gtc' in n \
                        else None
                except KeyError:
                    pass

            # Instrument id
            if not record['instrument_id']:
                try:
                    i = header['INSTRUME'].lower()
                    record[
                        'instrument_id'] = 1 if 'r-c spec' in i or 'test' in i or 'nod' in i else 2 if 'gmos-n' in i else 3 if 'gmos-s' in i else 4 if 'fors' in i else 5 if 'lris' in i else 6 if 'spex' in i else 7 if 'ldss3' in i else 8 if 'focas' in i else 9 if 'nirspec' in i else 10 if 'irs' in i else 11 if 'fire' in i else 12 if 'mage' in i else 13 if 'goldcam' in i else 14 if 'sinfoni' in i else 15 if 'osiris' in i else 16 if 'triplespec' in i else 17 if 'x-shooter' in i else 18 if 'gnirs' in i else 19 if 'wircam' in i else 20 if 'cormass' in i else 21 if 'isaac' in i else 22 if 'irac' in i else 23 if 'dis' in i else 24 if 'susi2' in i else 25 if 'ircs' in i else 26 if 'nirc' in i else 29 if 'stis' in i else 0
                except KeyError:
                    pass

    except:
        pass

    return record