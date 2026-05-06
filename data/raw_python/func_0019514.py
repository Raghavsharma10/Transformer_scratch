def _create_graphical_report(inputfile, pwm, background, closest_match, outdir, stats, best_id=None):
    """Create main gimme_motifs output html report."""
    if best_id is None:
        best_id = {}

    logger.debug("Creating graphical report")
    
    class ReportMotif(object):
        """Placeholder for motif stats."""
        pass

    config = MotifConfig()
    
    imgdir = os.path.join(outdir, "images")
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)
    
    motifs = read_motifs(pwm, fmt="pwm")
    
    roc_img_file = "%s_roc.%s"

    dbpwm = config.get_default_params()["motif_db"]
    pwmdir = config.get_motif_dir()

    dbmotifs = read_motifs(os.path.join(pwmdir, dbpwm), as_dict=True)
    
    report_motifs = []
    for motif in motifs:
        
        rm = ReportMotif()
        rm.id = motif.id
        rm.id_href = {"href": "#%s" % motif.id}
        rm.id_name = {"name": motif.id}
        rm.img = {"src":  os.path.join("images", "%s.png" % motif.id)}
        motif.to_img(os.path.join(outdir, "images/{}.png".format(motif.id)), fmt="PNG")
        
        # TODO: fix best ID
        rm.best = "Gimme"#best_id[motif.id]

        rm.consensus = motif.to_consensus()
        rm.stars = int(np.mean(
                [stats[str(motif)][bg].get("stars", 0) for bg in background]
                ) + 0.5)

        rm.bg = {}
        for bg in background:
            rm.bg[bg] = {}
            this_stats = stats.get(str(motif), {}).get(bg)
            # TODO: fix these stats
            rm.bg[bg]["e"] = "%0.2f" % this_stats.get("enr_at_fpr", 1.0)
            rm.bg[bg]["p"] = "%0.2f" % this_stats.get("phyper_at_fpr", 1.0)
            rm.bg[bg]["auc"] = "%0.3f" % this_stats.get("roc_auc", 0.5)
            rm.bg[bg]["mncp"] = "%0.3f" % this_stats.get("mncp", 1.0)
            rm.bg[bg]["roc_img"] = {"src": "images/" + os.path.basename(roc_img_file % (motif.id, bg)) + ".png"}
            rm.bg[bg][u"roc_img_link"] = {u"href": "images/" + os.path.basename(roc_img_file % (motif.id, bg)) + ".png"}

        rm.histogram_img = {"data":"images/%s_histogram.svg" % motif.id}
        rm.histogram_link= {"href":"images/%s_histogram.svg" % motif.id}
        
        match_id = closest_match[motif.id][0]
        dbmotifs[match_id].to_img(os.path.join(outdir, "images/{}.png".format(match_id)), fmt="PNG")
    
        rm.match_img = {"src":  "images/{}.png".format(match_id)}
        rm.match_id = closest_match[motif.id][0]
        rm.match_pval = "%0.2e" % closest_match[motif.id][1][-1]

        report_motifs.append(rm)
    
    total_report = os.path.join(outdir, "motif_report.html")

    star_img = os.path.join(config.get_template_dir(), "star.png")
    shutil.copyfile(star_img, os.path.join(outdir, "images", "star.png"))

    env = jinja2.Environment(loader=jinja2.FileSystemLoader([config.get_template_dir()]))
    template = env.get_template("report_template.jinja.html")
    # TODO: title
    result = template.render(
                    motifs=report_motifs, 
                    inputfile=inputfile, 
                    date=datetime.today().strftime("%d/%m/%Y"), 
                    version=__version__,
                    bg_types=list(background.keys()))

    with open(total_report, "wb") as f:
        f.write(result.encode('utf-8'))