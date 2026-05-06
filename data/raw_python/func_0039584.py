def v_posts_from_dataframe(df,N=1e4,alpha=0.23,l0=20,sigl=20):
    """ names: Prot, e_Prot, R, e_R, vsini, e_vsini
    """
    vsini_posts = []
    veq_posts = []
    if 'ep_R' in df:
        for R,dR_p,dR_m,P,dP,v,dv in zip(df['R'],df['ep_R'],df['em_R'],
                                  df['Prot'],df['e_Prot'],
                                  df['vsini'],df['e_vsini']):
            vsini_posts.append(stats.norm(v,dv))
            if dR_p==dR_m:
                veq_posts.append(Veq_Posterior(R,dR_p,P,dP))
            else:
                R_dist = dists.fit_doublegauss(R,dR_m,dR_p)
                Prot_dist = stats.norm(P,dP)
                veq_posts.append(Veq_Posterior_General(R_dist,Prot_dist,N=N,
                                                       l0=l0,sigl=sigl))
    else:
        for R,dR,P,dP,v,dv in zip(df['R'],df['e_R'],
                                  df['Prot'],df['e_Prot'],
                                  df['vsini'],df['e_vsini']):
            vsini_posts.append(stats.norm(v,dv))
            veq_posts.append(Veq_Posterior(R,dR,P,dP))
        
    return vsini_posts,veq_posts