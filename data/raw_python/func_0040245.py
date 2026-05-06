def triplifyGML(fname="foo.gml",fpath="./fb/",scriptpath=None,uid=None,sid=None,extra_info=None):
    """Produce a linked data publication tree from a standard GML file.

    INPUTS:
    => the file name (fname, with path) where the gdf file
    of the friendship network is.

    => the final path (fpath) for the tree of files to be created.
    
    => a path to the script that is calling this function (scriptpath).

    => the numeric id (uid) of the facebook user of which fname holds a friendship network 
    
    => the numeric id (sid) of the facebook user of which fname holds a friendship network 

    OUTPUTS:
    the tree in the directory fpath."""

#    aname=fname.split("/")[-1].split(".")[0]
    aname=fname.split("/")[-1].split(".")[0]
    if "RonaldCosta" in fname:
        aname=fname.split("/")[-1].split(".")[0]
        name,day,month,year=re.findall(".*/([a-zA-Z]*)(\d\d)(\d\d)(\d\d\d\d).gml",fname)[0]
        datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day)]).isoformat().split("T")[0]
        name_="Ronald Scherolt Costa"
    elif "AntonioAnzoategui" in fname:
        aname=re.findall(".*/([a-zA-Z]*\d*)",fname)[0]
        name,year,month,day,hour,minute=re.findall(r".*/([a-zA-Z]*).*_(\d+)_(\d*)_(\d*)_(\d*)_(\d*)_.*",fname)[0]
        datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day,hour,minute)]).isoformat()[:-3]
        name_="Antônio Anzoategui Fabbri"
    elif re.findall(".*/[a-zA-Z]*(\d)",fname):
        name,day,month,year=re.findall(".*/([a-zA-Z]*)(\d\d)(\d\d)(\d\d\d\d).*.gml",fname)[0]
        datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day)]).isoformat().split("T")[0]
        name_=" ".join(re.findall("[A-Z][^A-Z]*",name))
    elif re.findall("[a-zA-Z]*_",fname):
        name,year,month,day,hour,minute=re.findall(".*/([a-zA-Z]*).*(\d\d\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d).*.gml",fname)[0]
        datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day,hour,minute)]).isoformat().split("T")[0]
        name_=" ".join(re.findall("[A-Z][^A-Z]*",name))
    else:
        name_=" ".join(re.findall("[A-Z][^A-Z]*",name))
    aname+="_fb"
    name+="_fb"
    c("started snapshot",aname)
    tg=P.rdf.makeBasicGraph([["po","fb"],[P.rdf.ns.per,P.rdf.ns.fb]],"the {} facebook ego friendship network")
    tg2=P.rdf.makeBasicGraph([["po","fb"],[P.rdf.ns.per,P.rdf.ns.fb]],"RDF metadata for the facebook friendship network of my son")
    snapshot=P.rdf.IC([tg2],P.rdf.ns.po.FacebookSnapshot,
            aname,"Snapshot {}".format(aname))
    extra_uri=extra_val=[]
    if extra_info:
        extra_uri=[NS.po.extraInfo]
        extra_val=[extra_info]
    P.rdf.link([tg2],snapshot,"Snapshot {}".format(aname),
                          [P.rdf.ns.po.createdAt,
                          P.rdf.ns.po.triplifiedIn,
                          P.rdf.ns.po.donatedBy,
                          P.rdf.ns.po.availableAt,
                          P.rdf.ns.po.originalFile,
                          P.rdf.ns.po.onlineTranslateXMLFile,
                          P.rdf.ns.po.onlineTranslateTTLFile,
                          P.rdf.ns.po.translateXMLFile,
                          P.rdf.ns.po.translateTTLFile,
                           P.rdf.ns.po.onlineMetaXMLFile,
                           P.rdf.ns.po.onlineMetaTTLFile,
                           P.rdf.ns.po.metaXMLFilename,
                           P.rdf.ns.po.metaTTLFilename,
                          P.rdf.ns.po.acquiredThrough,
                          P.rdf.ns.rdfs.comment,
                          P.rdf.ns.fb.uid,
                          P.rdf.ns.fb.sid
                          ]+extra_uri,
                          [datetime_snapshot,
                           datetime.datetime.now(),
                           name,
                           "https://github.com/ttm/{}".format(aname),
                           "https://raw.githubusercontent.com/ttm/{}/master/base/{}".format(aname,fname.split("/")[-1]),
                           "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Translate.rdf".format(aname,aname),
                           "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Translate.ttl".format(aname,aname),
                           "{}Translate.rdf".format(aname),
                           "{}Translate.ttl".format(aname),
                            "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Meta.rdf".format(aname,aname),
                                "https://raw.githubusercontent.com/ttm/{}/master/rdf/{}Meta.ttl".format(aname,aname),
                                "{}Meta.owl".format(aname),
                                "{}Meta.ttl".format(aname),
                           "Netvizz",
                                "The facebook friendship network from {}".format(name_),
                                uid,
                                sid
                           ]+extra_val)
    #for friend_attr in fg2["friends"]:
    c((aname,name_,datetime_snapshot))
    fg2=x.read_gml(fname)
    c("read gml")
    for uid in fg2:
        c(uid)
        ind=P.rdf.IC([tg],P.rdf.ns.fb.Participant,"{}-{}".format(aname,uid))
        if "locale" in fg2.node[uid].keys():
            data=[fg2.node[uid][attr] for attr in ("id","label","locale","sex","agerank","wallcount")]
            uris=[NS.fb.gid,    NS.fb.name,
                  NS.fb.locale, NS.fb.sex,
                  NS.fb.agerank,NS.fb.wallcount]
        else:
            data=[fg2.node[uid][attr] for attr in ("id","label","sex","agerank","wallcount")]
            uris=[NS.fb.gid,    NS.fb.name,
                  NS.fb.sex,
                  NS.fb.agerank,NS.fb.wallcount]
        P.rdf.link([tg],ind, None,uris,data,draw=False)
        P.rdf.link_([tg],ind,None,[NS.po.snapshot],[snapshot],draw=False)


    #friends_=[fg2["friends"][i] for i in ("name","label","locale","sex","agerank")]
    #for name,label,locale,sex,agerank in zip(*friends_):
    #    ind=P.rdf.IC([tg],P.rdf.ns.fb.Participant,name,label)
    #    P.rdf.link([tg],ind,label,[P.rdf.ns.fb.uid,P.rdf.ns.fb.name,
    #                    P.rdf.ns.fb.locale,P.rdf.ns.fb.sex,
    #                    P.rdf.ns.fb.agerank],
    #                    [name,label,locale,sex,agerank])

    c("escritos participantes")
    #friendships_=[fg2["friendships"][i] for i in ("node1","node2")]
    i=1
    for uid1,uid2 in fg2.edges():
        flabel="{}-{}-{}".format(aname,uid1,uid2)
        ind=P.rdf.IC([tg],P.rdf.ns.fb.Friendship,flabel)
        uids=[P.rdf.IC(None,P.rdf.ns.fb.Participant,"{}-{}".format(aname,i)) for i in (uid1,uid2)]
        P.rdf.link_([tg],ind,flabel,[NS.po.snapshot]+[NS.fb.member]*2,
                                    [snapshot]+uids,draw=False)
        P.rdf.L_([tg],uids[0],P.rdf.ns.fb.friend,uids[1])
        if (i%1000)==0:
            c(i)
        i+=1
    c("escritas amizades")
    tg_=[tg[0]+tg2[0],tg[1]]
    fpath_="{}/{}/".format(fpath,aname)
    P.rdf.writeAll(tg_,aname+"Translate",fpath_,False,1)
    # copia o script que gera este codigo
    if not os.path.isdir(fpath_+"scripts"):
        os.mkdir(fpath_+"scripts")
    #shutil.copy(this_dir+"/../tests/rdfMyFNetwork2.py",fpath+"scripts/")
    shutil.copy(scriptpath,fpath_+"scripts/")
    # copia do base data
    if not os.path.isdir(fpath_+"base"):
        os.mkdir(fpath_+"base")
    shutil.copy(fname,fpath_+"base/")
    P.rdf.writeAll(tg2,aname+"Meta",fpath_,False)
    # faz um README
    with open(fpath_+"README","w") as f:
        f.write("""This repo delivers RDF data from the facebook
friendship network of {} ({}) collected at {}.
It has {} friends with metadata {};
and {} friendships.
The linked data is available at rdf/ dir and was
generated by the routine in the script/ directory.
Original data from Netvizz in data/\n""".format(
            name_,aname,datetime_snapshot,
            fg2.number_of_nodes(),
                    "name, locale (maybe), sex, agerank and wallcount",
                    fg2.number_of_edges()))