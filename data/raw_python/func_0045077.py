def rephase_pmns_standard(Unu, UeL, UeR):
    """Function to rephase the lepton rotation matrices in order to
    obtain the PMNS matrix in standard parametrization.

    The input matrices are assumed to diagonalize the charged lepton and
    neutrino mass matrices like

    ```
    UeL.conj().T @ Me @ UeR = Me_diag
    Unu.T @ Mnu @ Unu = Mnu_diag
    ```

    The PMNS matrix is given as `UPMNS = UeL.conj().T @ Unu`.

    Returns a tuple with the rephased versions of the input matrices.
    """
    U = UeL.conj().T @ Unu
    f = mixing_phases(U)
    Fdelta = np.diag(np.exp([1j*f['delta1'], 1j*f['delta2'], 1j*f['delta3']]))
    return Unu, UeL @ Fdelta, UeR @ Fdelta