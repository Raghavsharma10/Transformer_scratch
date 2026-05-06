def _flux(t, n, duration, std, offs=1):
    '''
    returns Gaussian shaped signal fluctuations [events]
    
    t --> times to calculate event for
    n --> numbers of events per sec
    duration --> event duration per sec
    std --> std of event if averaged over time
    offs --> event offset
    '''
    duration *= len(t) / t[-1]
    duration = int(max(duration, 1))

    pos = []
    n *= t[-1]
    pp = np.arange(len(t))
    valid = np.ones_like(t, dtype=bool)
    for _ in range(int(round(n))):
        try:
            ppp = np.random.choice(pp[valid], 1)[0]
            pos.append(ppp)
            valid[max(0, ppp - duration - 1):ppp + duration + 1] = False
        except ValueError:
            break
    sign = np.random.randint(0, 2, len(pos))
    sign[sign == 0] = -1

    out = np.zeros_like(t)

    amps = np.random.normal(loc=0, scale=1, size=len(pos))

    if duration > 2:
        evt = gaussian(duration, duration)
        evt -= evt[0]
    else:
        evt = np.ones(shape=duration)


    for s, p, a in zip(sign, pos, amps):
        pp = duration
        if p + duration > len(out):
            pp = len(out) - p

        out[p:p + pp] = s * a * evt[:pp]

    out /= out.std() / std
    out += offs
    return out