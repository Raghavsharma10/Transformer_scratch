def wget_files():
    """ Pull the files from the LAMOST archive """
    for f in lamost_id:
        short = (f.split('-')[2]).split('_')[0]
        filename = "%s/%s.gz" %(short,f)
        DIR = "/Users/annaho/Data/Li_Giants/Spectra_APOKASC"
        searchfor = "%s/%s.gz" %(DIR,f)
        if glob.glob(searchfor):
            print("done")
        else:
            #print(searchfor)
            os.system(
                    "wget http://dr2.lamost.org/sas/fits/%s" %(filename))
            new_filename = filename.split("_")[0] + "_" + filename.split("_")[2]
            os.system(
                    "wget http://dr2.lamost.org/sas/fits/%s" %(new_filename))