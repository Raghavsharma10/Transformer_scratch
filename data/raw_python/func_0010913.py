def make_label(loss, key):
    '''Create a legend label for an optimization run.'''
    algo, rate, mu, half, reg = key
    slots, args = ['{:.3f}', '{}', 'm={:.3f}'], [loss, algo, mu]
    if algo in 'SGD NAG RMSProp Adam ESGD'.split():
        slots.append('lr={:.2e}')
        args.append(rate)
    if algo in 'RMSProp ADADELTA ESGD'.split():
        slots.append('rmsh={}')
        args.append(half)
        slots.append('rmsr={:.2e}')
        args.append(reg)
    return ' '.join(slots).format(*args)