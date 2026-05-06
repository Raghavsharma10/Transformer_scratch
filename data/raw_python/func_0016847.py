def _construct_tensorflow_expression(slvr_cfg, feed_data, device, shard):
    """ Constructs a tensorflow expression for computing the RIME """
    zero = tf.constant(0)
    src_count = zero
    src_ph_vars = feed_data.src_ph_vars

    LSA = feed_data.local

    polarisation_type = slvr_cfg['polarisation_type']

    # Pull RIME inputs out of the feed staging_area
    # of the relevant shard, adding the feed once
    # inputs to the dictionary
    D = LSA.feed_many[shard].get_to_attrdict()
    D.update({k: fo.var for k, fo in LSA.feed_once.iteritems()})

    with tf.device(device):
        # Infer chunk dimensions
        model_vis_shape = tf.shape(D.model_vis)
        ntime, nbl, nchan, npol = [model_vis_shape[i] for i in range(4)]

        # Infer float and complex type
        FT, CT = D.uvw.dtype, D.model_vis.dtype

        # Compute sine and cosine of parallactic angles
        # for the beam
        beam_sin, beam_cos = rime.parallactic_angle_sin_cos(
                                        D.parallactic_angles)

        # Compute sine and cosine of feed rotation angle
        feed_sin, feed_cos = rime.parallactic_angle_sin_cos(
                                        D.parallactic_angles[:, :] +
                                        D.feed_angles[None, :])

        # Compute feed rotation
        feed_rotation = rime.feed_rotation(feed_sin, feed_cos, CT=CT,
                                           feed_type=polarisation_type)

    def antenna_jones(lm, stokes, alpha, ref_freq):
        """
        Compute the jones terms for each antenna.

        lm, stokes and alpha are the source variables.
        """

        # Compute the complex phase
        cplx_phase = rime.phase(lm, D.uvw, D.frequency, CT=CT)

        # Check for nans/infs in the complex phase
        phase_msg = ("Check that '1 - l**2  - m**2 >= 0' holds "
                    "for all your lm coordinates. This is required "
                    "for 'n = sqrt(1 - l**2 - m**2) - 1' "
                    "to be finite.")

        phase_real = tf.check_numerics(tf.real(cplx_phase), phase_msg)
        phase_imag = tf.check_numerics(tf.imag(cplx_phase), phase_msg)

        # Compute the square root of the brightness matrix
        # (as well as the sign)
        bsqrt, sgn_brightness = rime.b_sqrt(stokes, alpha,
            D.frequency, ref_freq, CT=CT,
            polarisation_type=polarisation_type)

        # Check for nans/infs in the bsqrt
        bsqrt_msg = ("Check that your stokes parameters "
                    "satisfy I**2 >= Q**2 + U**2 + V**2. "
                    "Montblanc performs a cholesky decomposition "
                    "of the brightness matrix and the above must "
                    "hold for this to produce valid values.")

        bsqrt_real = tf.check_numerics(tf.real(bsqrt), bsqrt_msg)
        bsqrt_imag = tf.check_numerics(tf.imag(bsqrt), bsqrt_msg)

        # Compute the direction dependent effects from the beam
        ejones = rime.e_beam(lm, D.frequency,
            D.pointing_errors, D.antenna_scaling,
            beam_sin, beam_cos,
            D.beam_extents, D.beam_freq_map, D.ebeam)

        deps = [phase_real, phase_imag, bsqrt_real, bsqrt_imag]
        deps = [] # Do nothing for now

        # Combine the brightness square root, complex phase,
        # feed rotation and beam dde's
        with tf.control_dependencies(deps):
            antenna_jones = rime.create_antenna_jones(bsqrt, cplx_phase,
                                                    feed_rotation, ejones, FT=FT)
            return antenna_jones, sgn_brightness

    # While loop condition for each point source type
    def point_cond(coherencies, npsrc, src_count):
        return tf.less(npsrc, src_ph_vars.npsrc)

    def gaussian_cond(coherencies, ngsrc, src_count):
        return tf.less(ngsrc, src_ph_vars.ngsrc)

    def sersic_cond(coherencies, nssrc, src_count):
        return tf.less(nssrc, src_ph_vars.nssrc)

    # While loop bodies
    def point_body(coherencies, npsrc, src_count):
        """ Accumulate visiblities for point source batch """
        S = LSA.sources['npsrc'][shard].get_to_attrdict()

        # Maintain source counts
        nsrc = tf.shape(S.point_lm)[0]
        src_count += nsrc
        npsrc +=  nsrc

        ant_jones, sgn_brightness = antenna_jones(S.point_lm,
            S.point_stokes, S.point_alpha, S.point_ref_freq)
        shape = tf.ones(shape=[nsrc,ntime,nbl,nchan], dtype=FT)
        coherencies = rime.sum_coherencies(D.antenna1, D.antenna2,
            shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, npsrc, src_count

    def gaussian_body(coherencies, ngsrc, src_count):
        """ Accumulate coherencies for gaussian source batch """
        S = LSA.sources['ngsrc'][shard].get_to_attrdict()

        # Maintain source counts
        nsrc = tf.shape(S.gaussian_lm)[0]
        src_count += nsrc
        ngsrc += nsrc

        ant_jones, sgn_brightness = antenna_jones(S.gaussian_lm,
            S.gaussian_stokes, S.gaussian_alpha, S.gaussian_ref_freq)
        gauss_shape = rime.gauss_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, S.gaussian_shape)
        coherencies = rime.sum_coherencies(D.antenna1, D.antenna2,
            gauss_shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, ngsrc, src_count

    def sersic_body(coherencies, nssrc, src_count):
        """ Accumulate coherencies for sersic source batch """
        S = LSA.sources['nssrc'][shard].get_to_attrdict()

        # Maintain source counts
        nsrc = tf.shape(S.sersic_lm)[0]
        src_count += nsrc
        nssrc += nsrc

        ant_jones, sgn_brightness = antenna_jones(S.sersic_lm,
            S.sersic_stokes, S.sersic_alpha, S.sersic_ref_freq)
        sersic_shape = rime.sersic_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, S.sersic_shape)
        coherencies = rime.sum_coherencies(D.antenna1, D.antenna2,
            sersic_shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, nssrc, src_count

    with tf.device(device):
        base_coherencies = tf.zeros(shape=[ntime,nbl,nchan,npol], dtype=CT)

        # Evaluate point sources
        summed_coherencies, npsrc, src_count = tf.while_loop(
            point_cond, point_body,
            [base_coherencies, zero, src_count])

        # Evaluate gaussians
        summed_coherencies, ngsrc, src_count = tf.while_loop(
            gaussian_cond, gaussian_body,
            [summed_coherencies, zero, src_count])

        # Evaluate sersics
        summed_coherencies, nssrc, src_count = tf.while_loop(
            sersic_cond, sersic_body,
            [summed_coherencies, zero, src_count])

        # Post process visibilities to produce model visibilites and chi squared
        model_vis, chi_squared = rime.post_process_visibilities(
            D.antenna1, D.antenna2, D.direction_independent_effects, D.flag,
            D.weight, D.model_vis, summed_coherencies, D.observed_vis)

    # Create enstaging_area operation
    put_op = LSA.output.put_from_list([D.descriptor, model_vis, chi_squared])

    # Return descriptor and enstaging_area operation
    return D.descriptor, put_op