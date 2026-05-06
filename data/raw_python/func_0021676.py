def cone(cls, **kwargs):
        """ Returns a cone.
            
            Kwargs:
                start (list): Start of cone, default [0, -1, 0].
                
                end (list): End of cone, default [0, 1, 0].
                
                radius (float): Maximum radius of cone at start, default 1.0.
                
                slices (int): Number of slices, default 16.
        """
        s = kwargs.get('start', Vector(0.0, -1.0, 0.0))
        e = kwargs.get('end', Vector(0.0, 1.0, 0.0))
        if isinstance(s, list):
            s = Vector(*s)
        if isinstance(e, list):
            e = Vector(*e)
        r = kwargs.get('radius', 1.0)
        slices = kwargs.get('slices', 16)
        ray = e.minus(s)
        
        axisZ = ray.unit()
        isY = (math.fabs(axisZ.y) > 0.5)
        axisX = Vector(float(isY), float(not isY), 0).cross(axisZ).unit()
        axisY = axisX.cross(axisZ).unit()
        startNormal = axisZ.negated()
        start = Vertex(s, startNormal)
        polygons = []
        
        taperAngle = math.atan2(r, ray.length())
        sinTaperAngle = math.sin(taperAngle)
        cosTaperAngle = math.cos(taperAngle)
        def point(angle):
            # radial direction pointing out
            out = axisX.times(math.cos(angle)).plus(
                axisY.times(math.sin(angle)))
            pos = s.plus(out.times(r))
            # normal taking into account the tapering of the cone
            normal = out.times(cosTaperAngle).plus(axisZ.times(sinTaperAngle))
            return pos, normal

        dt = math.pi * 2.0 / float(slices)
        for i in range(0, slices):
            t0 = i * dt
            i1 = (i + 1) % slices
            t1 = i1 * dt
            # coordinates and associated normal pointing outwards of the cone's
            # side
            p0, n0 = point(t0)
            p1, n1 = point(t1)
            # average normal for the tip
            nAvg = n0.plus(n1).times(0.5)
            # polygon on the low side (disk sector)
            polyStart = Polygon([start.clone(), 
                                 Vertex(p0, startNormal), 
                                 Vertex(p1, startNormal)])
            polygons.append(polyStart)
            # polygon extending from the low side to the tip
            polySide = Polygon([Vertex(p0, n0), Vertex(e, nAvg), Vertex(p1, n1)])
            polygons.append(polySide)

        return CSG.fromPolygons(polygons)