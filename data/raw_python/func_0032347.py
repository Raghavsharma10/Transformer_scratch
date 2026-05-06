def ticket1to2(old):
    """
    change Ticket to refer to Products and not benefactor factories.
    """
    if isinstance(old.benefactor, Multifactor):
        types = list(chain(*[b.powerupNames for b in
                old.benefactor.benefactors('ascending')]))
    elif isinstance(old.benefactor, InitializerBenefactor):
        #oh man what a mess
        types = list(chain(*[b.powerupNames for b in
                old.benefactor.realBenefactor.benefactors('ascending')]))
    newProduct = old.store.findOrCreate(Product,
                                        types=types)

    if old.issuer is None:
        issuer = old.store.findOrCreate(TicketBooth)
    else:
        issuer = old.issuer
    t = old.upgradeVersion(Ticket.typeName, 1, 2,
                           product = newProduct,
                           issuer = issuer,
                           booth = old.booth,
                           avatar = old.avatar,
                           claimed = old.claimed,

                           email = old.email,
                           nonce = old.nonce)