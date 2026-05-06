def calc_surfdist(surface, labels, annot, reg, origin, target):
  import nibabel as nib
  import numpy as np
  import os
  from surfdist import load, utils, surfdist
  import csv

  """ inputs:
  surface - surface file (e.g. lh.pial, with full path)
  labels - label file (e.g. lh.cortex.label, with full path)
  annot - annot file (e.g. lh.aparc.a2009s.annot, with full path)
  reg - registration file (lh.sphere.reg)
  origin - the label from which we calculate distances
  target - target surface (e.g. fsaverage4)
  """

  # Load stuff
  surf = nib.freesurfer.read_geometry(surface)
  cort = np.sort(nib.freesurfer.read_label(labels))
  src  = load.load_freesurfer_label(annot, origin, cort)

  # Calculate distances
  dist = surfdist.dist_calc(surf, cort, src)

  # Project distances to target
  trg = nib.freesurfer.read_geometry(target)[0]
  native = nib.freesurfer.read_geometry(reg)[0]
  idx_trg_to_native = utils.find_node_match(trg, native)[0]

  # Get indices in trg space 
  distt = dist[idx_trg_to_native]
  
  # Write to file and return file handle
  filename = os.path.join(os.getcwd(),'distances.csv')
  distt.tofile(filename,sep=",")

  return filename