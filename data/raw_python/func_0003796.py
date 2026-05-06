def random_orthonormal(normal):
    """Return a random normalized vector orthogonal to the given vector"""
    u = normal_fns[np.argmin(np.fabs(normal))](normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    alpha = np.random.uniform(0.0, np.pi*2)
    return np.cos(alpha)*u + np.sin(alpha)*v