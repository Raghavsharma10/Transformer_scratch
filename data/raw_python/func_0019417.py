def _kernelized_dist2centers(K, n_clusters, wmemb, kernel_dist):
    """ Computin the distance in transformed feature space to 
         cluster centers.
 
        K is the kernel gram matrix.
        wmemb contains cluster assignment. {0,1}

        Assume j is the cluster id:
        ||phi(x_i) - Phi_center_j|| = K_ii - 2 sum w_jh K_ih + 
                                      sum_r sum_s w_jr w_js K_rs
    """
    n_samples = K.shape[0]
    
    for j in range(n_clusters):
        memb_j = np.where(wmemb == j)[0]
        size_j = memb_j.shape[0]

        K_sub_j = K[memb_j][:, memb_j]
         
        kernel_dist[:,j] = 1 + np.sum(K_sub_j) /(size_j*size_j)
        kernel_dist[:,j] -= 2 * np.sum(K[:, memb_j], axis=1) / size_j

    return