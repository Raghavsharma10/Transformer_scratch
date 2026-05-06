def triplifyGDFInteraction(fname="foo.gdf",fpath="./fb/",scriptpath=None,uid=None,sid=None,dlink=None):
    """Produce a linked data publication tree from GDF files of a Facebook interaction network.

    INPUTS:
    => the file name (fname, with path) where the gdf file
    of the friendship network is.

    => the final path (fpath) for the tree of files to be created.
    
    => a path to the script that is calling this function (scriptpath).

    => the numeric id (uid) of the facebook group
    
    => the string id (sid) of the facebook group of which fname holds a friendship network 

    OUTPUTS:
    the tree in the directory fpath."""
    #aname=fname.split("/")[-1].split(".")[0]+"_fb"
    aname=fname.split("/")[-1].split(".")[0]
    if re.findall("[a-zA-Z]*_[0-9]",fname):
        name,year,month,day,hour,minute=re.findall(".*/([a-zA-Z]*).*(\d\d\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d).*.gdf",fname)[0]
        datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day,hour,minute)]).isoformat().split("T")[0]
        name_=" ".join(re.findall("[A-Z][^A-Z]*",name))
    elif re.findall("(\d)",fname):
        name,day,month,year=re.findall(".*/([a-zA-Z]*)(\d\d)(\d\d)(\d\d\d\d).*.gdf",fname)[0]
        datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day)]).isoformat().split("T")[0]
        name_=" ".join(re.findall("[A-Z][^A-Z]*",name))
    else:
        datetime_snapshot=datetime.datetime(2013,3,15).isoformat().split("T")[0]
        name_=" ".join(re.findall("[A-Z][^A-Z]*",aname))
    aname+="_fb"
    name=aname


    tg=P.rdf.makeBasicGraph([["po","fb"],[P.rdf.ns.per,P.rdf.ns.fb]],"The facebook interaction network from the {} file".format(fname)) # drop de agraph
    tg2=P.rdf.makeBasicGraph([["po"],[P.rdf.ns.per]],"Metadata for my facebook ego friendship network RDF files") # drop de agraph
    ind=P.rdf.IC([tg2],P.rdf.ns.po.Snapshot,
            aname,"Snapshot {}".format(aname))

    foo={"uris":[],"vals":[]}
    if sid:
        foo["uris"].append(P.rdf.ns.fb.sid)
        foo["vals"].append(sid)
    if uid:
        foo["uris"].append(P.rdf.ns.fb.uid)
        foo["vals"].append(uid)
    if dlink:
        foo["uris"].append(P.rdf.ns.fb.link)
        foo["vals"].append(dlink)
    P.rdf.link([tg2],ind,"Snapshot {}".format(aname),
                        [P.rdf.ns.po.createdAt,
                          P.rdf.ns.po.triplifiedIn,
                          P.rdf.ns.po.donatedBy,
                          P.rdf.ns.po.availableAt,
                          P.rdf.ns.po.originalFile,
                          P.rdf.ns.po.rdfFile,
                          P.rdf.ns.po.ttlFile,
                          P.rdf.ns.po.discorveryRDFFile,
                          P.rdf.ns.po.discoveryTTLFile,
                          P.rdf.ns.po.acquiredThrough,
                          P.rdf.ns.rdfs.comment,
                          ]+foo["uris"],
                          [datetime_snapshot,
                           datetime.datetime.now(),
                           name,
                           "https://github.com/ttm/{}".format(aname),
                           "https://raw.githubusercontent.com/ttm/{}/master/base/{}".format(aname,fname.split("/")),
                           "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Translate.owl".format(aname,aname),
                           "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Translate.ttl".format(aname,aname),
                                "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Meta.owl".format(aname,aname),
                                "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Meta.ttl".format(aname,aname),
                           "Netvizz",
                                "The facebook friendship network from {}".format(name_),
                           ]+foo["vals"])
    #for friend_attr in fg2["friends"]:
    fg2=readGDF(fname)
    tkeys=list(fg2["friends"].keys())
    def trans(tkey):
        if tkey=="name":
            return "uid"
        if tkey=="label":
            return "name"
        return tkey
    foo={"uris":[],"vals":[]}
    for tkey in tkeys:
        if tkey=="groupid":
            P.rdf.link([tg2],ind,"Snapshot {}".format(aname),
                        [P.rdf.ns.po.uid,],
                        [fg2["friends"][tkey][0]])
        if tkey:
            foo["uris"]+=[eval("P.rdf.ns.fb."+trans(tkey))]
            foo["vals"]+=[fg2["friends"][tkey]]
    print(tkeys)
    iname=tkeys.index("name")
    ilabel=tkeys.index("label")
    icount=0
    name_label={}
    for vals_ in zip(*foo["vals"]):
        name,label=[foo["vals"][i][icount] for i in (iname,ilabel)]
        if not label:
            label="po:noname"
            vals_=list(vals_)
            vals_[ilabel]=label
        name_label[name]=label
        ind=P.rdf.IC([tg],P.rdf.ns.fb.Participant,name,label)
        P.rdf.link([tg],ind,label,foo["uris"],
                        vals_,draw=False)
        icount+=1

    friendships_=[fg2["friendships"][i] for i in ("node1","node2")]
    c("escritos participantes")
    i=1
    for uid1,uid2 in zip(*friendships_):
        flabel="{}-{}".format(uid1,uid2)
        labels=[name_label[uu] for uu in (uid1,uid2)]
        ind=P.rdf.IC([tg],P.rdf.ns.fb.Friendship,
                flabel)
                #flabel,"Friendship "+flabel)
        ind1=P.rdf.IC(None,P.rdf.ns.fb.Participant,uid1)
        ind2=P.rdf.IC(None,P.rdf.ns.fb.Participant,uid2)
        uids=[r.URIRef(P.rdf.ns.fb.Participant+"#"+str(i)) for i in (uid1,uid2)]
        P.rdf.link_([tg],ind,"Friendship "+flabel,[P.rdf.ns.fb.member]*2,
                            uids,labels,draw=False)
        P.rdf.L_([tg],uids[0],P.rdf.ns.fb.friend,uids[1])
        if (i%1000)==0:
            c(i)
        i+=1
    P.rdf.G(tg[0],P.rdf.ns.fb.friend,
            P.rdf.ns.rdf.type,
            P.rdf.ns.owl.SymmetricProperty)
    c("escritas amizades")
    tg_=[tg[0]+tg2[0],tg[1]]
    fpath_="{}{}/".format(fpath,aname)
    P.rdf.writeAll(tg_,aname+"Translate",fpath_,False,1)
    # copia o script que gera este codigo
    if not os.path.isdir(fpath_+"scripts"):
        os.mkdir(fpath_+"scripts")
    shutil.copy(scriptpath,fpath_+"scripts/")
    # copia do base data
    if not os.path.isdir(fpath_+"base"):
        os.mkdir(fpath_+"base")
    shutil.copy(fname,fpath_+"base/")
    P.rdf.writeAll(tg2,aname+"Meta",fpath_,1)
    # faz um README
    with open(fpath_+"README","w") as f:
        f.write("""This repo delivers RDF data from the facebook
friendship network of {} collected at {}.
It has {} friends with metadata {};
and {} friendships.
The linked data is available at rdf/ dir and was
generated by the routine in the script/ directory.
Original data from Netvizz in data/\n""".format(
            name_,datetime_snapshot,
            len(fg2["friends"]["name"]),
                    "facebook numeric id, name, locale, sex and agerank",
                    len(fg2["friendships"]["node1"])
                    ))