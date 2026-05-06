def runDia(diagram):
    """Generate the diagrams using Dia."""
    ifname = '{}.dia'.format(diagram)
    ofname = '{}.png'.format(diagram)
    cmd = 'dia -t png-libart -e {} {}'.format(ofname, ifname)
    print('  {}'.format(cmd))
    subprocess.call(cmd, shell=True)
    return True