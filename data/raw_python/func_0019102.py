def interp_qa_v1(self):
    """Calculate the lake outflow based on linear interpolation.

    Required control parameters:
      |N|
      |llake_control.Q|

    Required derived parameters:
      |llake_derived.TOY|
      |llake_derived.VQ|

    Required aide sequence:
      |llake_aides.VQ|

    Calculated aide sequence:
      |llake_aides.QA|

    Examples:

        In preparation for the following examples, define a short simulation
        time period with a simulation step size of 12 hours and initialize
        the required model object:

        >>> from hydpy import pub
        >>> pub.timegrids = '2000.01.01','2000.01.04', '12h'
        >>> from hydpy.models.llake import *
        >>> parameterstep()

        Next, for the sake of brevity, define a test function:

        >>> def test(*vqs):
        ...     for vq in vqs:
        ...         aides.vq(vq)
        ...         model.interp_qa_v1()
        ...         print(repr(aides.vq), repr(aides.qa))

        The following three relationships between the auxiliary term `vq` and
        the tabulated discharge `q` are taken as examples.  Each one is valid
        for one of the first three days in January and is defined via five
        nodes:

        >>> n(5)
        >>> derived.toy.update()
        >>> derived.vq(_1_1_6=[0., 1., 2., 2., 3.],
        ...            _1_2_6=[0., 1., 2., 2., 3.],
        ...            _1_3_6=[0., 1., 2., 3., 4.])
        >>> q(_1_1_6=[0., 0., 0., 0., 0.],
        ...   _1_2_6=[0., 2., 5., 6., 9.],
        ...   _1_3_6=[0., 2., 1., 3., 2.])

        In the first example, discharge does not depend on the actual value
        of the auxiliary term and is always zero:

        >>> model.idx_sim = pub.timegrids.init['2000.01.01']
        >>> test(0., .75, 1., 4./3., 2., 7./3., 3., 10./3.)
        vq(0.0) qa(0.0)
        vq(0.75) qa(0.0)
        vq(1.0) qa(0.0)
        vq(1.333333) qa(0.0)
        vq(2.0) qa(0.0)
        vq(2.333333) qa(0.0)
        vq(3.0) qa(0.0)
        vq(3.333333) qa(0.0)

        The seconds example demonstrates that relationships are allowed to
        contain jumps, which is the case for the (`vq`,`q`) pairs (2,6) and
        (2,7).  Also it demonstrates that when the highest `vq` value is
        exceeded linear extrapolation based on the two highest (`vq`,`q`)
        pairs is performed:

        >>> model.idx_sim = pub.timegrids.init['2000.01.02']
        >>> test(0., .75, 1., 4./3., 2., 7./3., 3., 10./3.)
        vq(0.0) qa(0.0)
        vq(0.75) qa(1.5)
        vq(1.0) qa(2.0)
        vq(1.333333) qa(3.0)
        vq(2.0) qa(5.0)
        vq(2.333333) qa(7.0)
        vq(3.0) qa(9.0)
        vq(3.333333) qa(10.0)

        The third example shows that the relationships do not need to be
        arranged monotonously increasing.  Particualarly for the extrapolation
        range, this could result in negative values of `qa`, which is avoided
        by setting it to zero in such cases:

        >>> model.idx_sim = pub.timegrids.init['2000.01.03']
        >>> test(.5, 1.5, 2.5, 3.5, 4.5, 10.)
        vq(0.5) qa(1.0)
        vq(1.5) qa(1.5)
        vq(2.5) qa(2.0)
        vq(3.5) qa(2.5)
        vq(4.5) qa(1.5)
        vq(10.0) qa(0.0)

    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    aid = self.sequences.aides.fastaccess
    idx = der.toy[self.idx_sim]
    for jdx in range(1, con.n):
        if der.vq[idx, jdx] >= aid.vq:
            break
    aid.qa = ((aid.vq-der.vq[idx, jdx-1]) *
              (con.q[idx, jdx]-con.q[idx, jdx-1]) /
              (der.vq[idx, jdx]-der.vq[idx, jdx-1]) +
              con.q[idx, jdx-1])
    aid.qa = max(aid.qa, 0.)