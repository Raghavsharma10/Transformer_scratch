def read_noise_curve(noise_curve, noise_type_in='ASD', noise_type_out='ASD',
                     add_wd_noise=False, wd_noise='HB_wd_noise', wd_noise_type_in='ASD'):
    """Simple auxillary function that can read noise curves in.

    This function can read in noise curves from a provided file or those that are preinstalled
    with this installation. All pre-installed noise curves are in the form of
    an amplitude spectral density. Information on each one is found in each specific file.
    These are located in the `noise_curves` folder.

    Pre-installed really just means in the noise_curves folder. Therefore, curves can be added
    and called with only a string.

    Arguments:
        noise_curve (str): Either a file path to a noise curve
            or a str represented pre-loaded sensitivity curve. If using pre-loaded curve,
            choices are LPA (LISA Phase A), PL (Proposed LISA), CL (Classic LISA),
            CLLF (Classic LISA Low Frequency), PLCS (Proposed LISA Constant Slope),
            or PLHB (Proposed LISA Higher Break).
            See the arXiv paper above for the meaning behind each choice and a plot with each curve.
        noise_type_in/noise_type_out (str, optional): Type of noise input/output.
            Choices are `ASD`, `PSD`, or `char_strain`. Default for both is `ASD`.
        add_wd_noise (bool, optional): If True, include wd noise.
        wd_noise (str, optional): File path to wd background noise or string representing
            those in the noise curves folder. Default is the Hiscock et al 2000 approximation
            of the Hils & Bender 1997 white dwarf background (`HB_wd_noise`).
        wd_noise_type_in (str, optional): Type of wd noise input.
            The output will be the same as ``noise_type_out``.
            Choices are `ASD`, `PSD`, or `char_strain`. Default for both is `ASD`.

        Returns:
            (tuple of arrays): Frequency and amplitude arrays of type ``noise_type_out``.

    """
    possible_noise_types = ['ASD', 'PSD', 'char_strain']
    if noise_type_in not in possible_noise_types:
        raise ValueError('noise_type_in must be either ASD, PSD, or char_strain.')
    if noise_type_out not in possible_noise_types:
        raise ValueError('noise_type_out must be either ASD, PSD, or char_strain.')

    # find the noise curve file
    if noise_curve[-4:] == '.txt':
        noise = ascii.read(noise_curve)
    else:
        cfd = os.path.dirname(os.path.abspath(__file__))
        noise = ascii.read(cfd + '/noise_curves/' + noise_curve + '.txt')

    # read it in
    f_n = np.asarray(noise['f'])
    amp_n = np.asarray(noise[noise_type_in])

    if noise_type_in != noise_type_out:
        amp_n = globals()[noise_type_in.lower() + '_to_' + noise_type_out.lower()](f_n, amp_n)

    # add wd_noise if true
    if add_wd_noise:
        if wd_noise_type_in not in possible_noise_types:
            raise ValueError('wd_noise_type_in must be either ASD, PSD, or char_strain.')

        if wd_noise[-4:] == '.txt':
            wd_data = ascii.read(wd_noise)
        else:
            cfd = os.path.dirname(os.path.abspath(__file__))
            wd_data = ascii.read(cfd + '/noise_curves/' + wd_noise + '.txt')

        f_n_wd = np.asarray(wd_data['f'])
        amp_n_wd = np.asarray(wd_data[wd_noise_type_in])

        if wd_noise_type_in != noise_type_out:
            amp_n_wd = globals()[noise_type_in.lower()
                                 + '_to_' + noise_type_out.lower()](f_n_wd, amp_n_wd)

        f_n, amp_n = combine_with_wd_noise(f_n, amp_n, f_n_wd, amp_n_wd)

    return f_n, amp_n