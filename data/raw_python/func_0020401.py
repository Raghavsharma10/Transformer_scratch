def show_G_distribution(data):
    '''Show the distribution of the G function.'''
    Xs, t = fitting.preprocess_data(data)  

    Theta, Phi = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 50))
    G = []

    for i in range(len(Theta)):
        G.append([])
        for j in range(len(Theta[i])):
            w = fitting.direction(Theta[i][j], Phi[i][j])
            G[-1].append(fitting.G(w, Xs))

    plt.imshow(G, extent=[0, np.pi, 0, 2 * np.pi], origin='lower')
    plt.show()