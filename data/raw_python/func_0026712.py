def flow2magang(flow, dtype=uint8):
    """
    flow dimensions y,x,2  3-D.  flow[...,0] is magnitude, flow[...,1] is angle
    """
    fx, fy = flow[..., 0], flow[..., 1]
    return hypot(fx, fy).astype(dtype), arctan2(fy, fx) + pi