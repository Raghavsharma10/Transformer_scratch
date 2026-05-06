def sphere(cls, **kwargs):
        """ Returns a sphere.
            
            Kwargs:
                center (list): Center of sphere, default [0, 0, 0].
                
                radius (float): Radius of sphere, default 1.0.
                
                slices (int): Number of slices, default 16.
                
                stacks (int): Number of stacks, default 8.
        """
        center = kwargs.get('center', [0.0, 0.0, 0.0])
        if isinstance(center, float):
            center = [center, center, center]
        c = Vector(center)
        r = kwargs.get('radius', 1.0)
        if isinstance(r, list) and len(r) > 2:
            r = r[0]
        slices = kwargs.get('slices', 16)
        stacks = kwargs.get('stacks', 8)
        polygons = []
        def appendVertex(vertices, theta, phi):
            d = Vector(
                math.cos(theta) * math.sin(phi),
                math.cos(phi),
                math.sin(theta) * math.sin(phi))
            vertices.append(Vertex(c.plus(d.times(r)), d))
            
        dTheta = math.pi * 2.0 / float(slices)
        dPhi = math.pi / float(stacks)

        j0 = 0
        j1 = j0 + 1
        for i0 in range(0, slices):
            i1 = i0 + 1
            #  +--+
            #  | /
            #  |/
            #  +
            vertices = []
            appendVertex(vertices, i0 * dTheta, j0 * dPhi)
            appendVertex(vertices, i1 * dTheta, j1 * dPhi)
            appendVertex(vertices, i0 * dTheta, j1 * dPhi)
            polygons.append(Polygon(vertices))

        j0 = stacks - 1
        j1 = j0 + 1
        for i0 in range(0, slices):
            i1 = i0 + 1
            #  +
            #  |\
            #  | \
            #  +--+
            vertices = []
            appendVertex(vertices, i0 * dTheta, j0 * dPhi)
            appendVertex(vertices, i1 * dTheta, j0 * dPhi)
            appendVertex(vertices, i0 * dTheta, j1 * dPhi)
            polygons.append(Polygon(vertices))
            
        for j0 in range(1, stacks - 1):
            j1 = j0 + 0.5
            j2 = j0 + 1
            for i0 in range(0, slices):
                i1 = i0 + 0.5
                i2 = i0 + 1
                #  +---+
                #  |\ /|
                #  | x |
                #  |/ \|
                #  +---+
                verticesN = []
                appendVertex(verticesN, i1 * dTheta, j1 * dPhi)
                appendVertex(verticesN, i2 * dTheta, j2 * dPhi)
                appendVertex(verticesN, i0 * dTheta, j2 * dPhi)
                polygons.append(Polygon(verticesN))
                verticesS = []
                appendVertex(verticesS, i1 * dTheta, j1 * dPhi)
                appendVertex(verticesS, i0 * dTheta, j0 * dPhi)
                appendVertex(verticesS, i2 * dTheta, j0 * dPhi)
                polygons.append(Polygon(verticesS))
                verticesW = []
                appendVertex(verticesW, i1 * dTheta, j1 * dPhi)
                appendVertex(verticesW, i0 * dTheta, j2 * dPhi)
                appendVertex(verticesW, i0 * dTheta, j0 * dPhi)
                polygons.append(Polygon(verticesW))
                verticesE = []
                appendVertex(verticesE, i1 * dTheta, j1 * dPhi)
                appendVertex(verticesE, i2 * dTheta, j0 * dPhi)
                appendVertex(verticesE, i2 * dTheta, j2 * dPhi)
                polygons.append(Polygon(verticesE))
                
        return CSG.fromPolygons(polygons)