def prepare_plot_data(data_file):
    """
    Return a list of Plotly elements representing the network graph
    """

    G = ig.Graph.Read_GML(data_file)

    layout = G.layout('graphopt')
    labels = list(G.vs['label'])

    N = len(labels)
    E = [e.tuple for e in G.es]

    community = G.community_multilevel().membership
    communities = len(set(community))

    color_list = community_colors(communities)

    Xn = [layout[k][0] for k in range(N)]
    Yn = [layout[k][1] for k in range(N)]

    Xe = []
    Ye = []

    for e in E:
        Xe += [layout[e[0]][0], layout[e[1]][0], None]
        Ye += [layout[e[0]][1], layout[e[1]][1], None]

    lines = Scatter(x=Xe,
                    y=Ye,
                    mode='lines',
                    line=Line(color='rgb(210,210,210)', width=1),
                    hoverinfo='none'
                    )
    plot_data = [lines]

    node_x = [[] for i in range(communities)]
    node_y = [[] for i in range(communities)]
    node_labels = [[] for i in range(communities)]

    for j in range(len(community)):
        index = community[j]

        node_x[index].append(layout[j][0])
        node_y[index].append(layout[j][1])
        node_labels[index].append(labels[j])

    for i in range(communities):
        trace = Scatter(x=node_x[i],
                        y=node_y[i],
                        mode='markers',
                        name='ntw',
                        marker=Marker(symbol='dot',
                                      size=5,
                                      color=color_list[i],
                                      line=Line(
                                          color='rgb(50,50,50)', width=0.5)
                                      ),
                        text=node_labels[i],
                        hoverinfo='text'
                        )

        plot_data.append(trace)

    return plot_data