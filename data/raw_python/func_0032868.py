def printChannelColRow(campaign, ra, dec):
    """Prints the channel, col, row for a given campaign and coordinate."""
    fovobj = fields.getKeplerFov(campaign)
    ch, col, row = fovobj.getChannelColRow(ra, dec)
    print("Position in C{}: channel {}, col {:.0f}, row {:.0f}.".format(campaign, int(ch), col, row))