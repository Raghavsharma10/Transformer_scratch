def rephase_standard(UuL, UdL, UuR, UdR):
    """Function to rephase the quark rotation matrices in order to
    obtain the CKM matrix in standard parametrization.

    The input matrices are assumed to diagonalize the up-type and down-type
    quark matrices like

    ```
    UuL.conj().T @ Mu @ UuR = Mu_diag
    UdL.conj().T @ Md @ UdR = Md_diag
    ```

    The CKM matrix is given as `VCKM = UuL.conj().T @ UdL`.

    Returns a tuple with the rephased versions of the input matrices.
    """
    K = UuL.conj().T @ UdL
    f = mixing_phases(K)
    Fdelta = np.diag(np.exp([1j*f['delta1'], 1j*f['delta2'], 1j*f['delta3']]))
    Fphi = np.diag(np.exp([-1j*f['phi1']/2., -1j*f['phi2']/2., 0]))
    return UuL @ Fdelta, UdL @ Fphi.conj(), UuR @ Fdelta, UdR @ Fphi.conj()