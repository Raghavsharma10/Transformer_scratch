def graph(self, fnm=None, size=None, fntsz=None, fntfm=None, clrgen=None,
              rmsz=False, prog='dot'):
        """
        Construct call graph

        Parameters
        ----------
        fnm : None or string, optional (default None)
          Filename of graph file to be written. File type is determined by
          the file extentions (e.g. dot for 'graph.dot' and SVG for
          'graph.svg'). If None, a file is not written.
        size : string or None, optional (default None)
          Graph image size specification string.
        fntsz : int or None, optional (default None)
          Font size for text.
        fntnm : string or None, optional (default None)
          Font family specification string.
        clrgen : function or None, optional (default None)
          Function to call to generate the group colours. This function
          should take an integer specifying the number of groups as an
          argument and return a list of graphviz-compatible colour
          specification strings.
        rmsz : bool, optional (default False)
          If True, remove the width and height specifications from an
          SVG format output file so that the size scales properly when
          viewed in a web browser
        prog : string, optional (default 'dot')
          Name of graphviz layout program to use.

        Returns
        -------
        pgr : pygraphviz.AGraph
          Call graph of traced function calls
        """

        # Default colour generation function
        if clrgen is None:
            clrgen = lambda n: self._clrgen(n, 0.330, 0.825)

        # Generate color list
        clrlst = clrgen(len(self.group))

        # Initialise a pygraphviz graph
        g = pgv.AGraph(strict=False, directed=True, landscape=False,
                       rankdir='LR', newrank=True, fontsize=fntsz,
                       fontname=fntfm, size=size, ratio='compress',
                       color='black', bgcolor='#ffffff00')
        # Set graph attributes
        g.node_attr.update(penwidth=0.25, shape='box', style='rounded,filled')

        # Iterate over functions adding them as graph nodes
        for k in self.fncts:
            g.add_node(k, fontsize=fntsz, fontname=fntfm)
            # If lnksub regex pair is provided, compute an href link
            # target from the node name and add it as an attribute to
            # the node
            if self.lnksub is not None:
                lnktgt = re.sub(self.lnksub[0], self.lnksub[1], k)
                g.get_node(k).attr.update(href=lnktgt, target="_top")
            # If function has no calls to it, set its rank to "source"
            if self.fncts[k][1] == 0:
                g.get_node(k).attr.update(rank='source')

        # If groups defined, construct a subgraph for each and add the
        # nodes in each group to the corresponding subgraph
        if self.group:
            fngrpnm = {}
            # Iterate over group number/group name pairs
            for k in zip(range(len(self.group)), sorted(self.group)):
                g.add_subgraph(self.group[k[1]], name='cluster_' + k[1],
                               label=k[1], penwidth=2, style='dotted',
                               pencolor=clrlst[k[0]])
                # Iterate over nodes in current group
                for l in self.group[k[1]]:
                    # Create record of function group number
                    fngrpnm[l] = k[0]
                    # Set common group colour for current node
                    g.get_node(l).attr.update(fillcolor=clrlst[k[0]])

        # Iterate over function calls, adding each as an edge
        for k in self.calls:
            # If groups defined, set edge colour according to group of
            # calling function, otherwise set a standard colour
            if self.group:
                g.add_edge(k[0], k[1], penwidth=2, color=clrlst[fngrpnm[k[0]]])
            else:
                g.add_edge(k[0], k[1], color='grey')

        # Call layout program
        g.layout(prog=prog)

        # Write graph file if filename provided
        if fnm is not None:
            ext = os.path.splitext(fnm)[1]
            if ext == '.dot':
                g.write(fnm)
            else:
                if ext == '.svg' and rmsz:
                    img = g.draw(format='svg').decode('utf-8')
                    cp = re.compile(r'\n<svg width=\"[^\"]*\" '
                                    'height=\"[^\"]*\"')
                    img = cp.sub(r'\n<svg', img, count=1)
                    with open(fnm, 'w') as fd:
                        fd.write(img)
                else:
                    g.draw(fnm)

        # Return graph object
        return g