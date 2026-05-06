def is_bgzf(filename):
  """Pre: filename to test if it is a bgzf format

  Post: True or False

  :param filename:
  :type filename: string
  :return: if its a bgzf
  :rtype: bool
  """
  with open(filename,'rb') as inf:
   bytes1 = inf.read(12)
   if len(bytes1) != 12:
     sys.stderr.write("File length ERROR\n")
     return False
   try:
     gzip_id1,gzip_id2,compression_method,flag,mtime,xfl,osval,xlen=struct.unpack('<BBBBIBBH',bytes1)
   except:
     sys.stderr.write("Unpack ERROR\n")
     return False
   if gzip_id1 != 31:
     sys.stderr.write("ID1 ERROR\n")
     return False
   if gzip_id2 != 139:
     sys.stderr.write("ID2 ERROR\n")
     return False
   if compression_method != 8:
     sys.stderr.write("Compression Method ERROR\n")
     return False
   if flag != 4:
     sys.stderr.write("flg ERROR\n")
     return False
   if xlen < 6:
     sys.stderr.write("no extra fields ERROR\n")
   # ready to look in extra field
   bytes2 = inf.read(xlen) # all the extra field stuff 
   if len(bytes2) != xlen: 
     sys.stderr.write("file length ERROR\n")
     return False
   s = StringIO.StringIO(bytes2)
   has_id = False
   obsslen = 0
   while True:
     v1 = s.read(4)
     if len(v1) == 0: break
     if len(v1) < 4:
       sys.stderr.write("lack header values ERROR\n")
       return False
     s1,s2,slen = struct.unpack('<BBH',v1)
     if s1 == 66 and s2 == 67:
       has_id = True
       obsslen = slen
     v = s.read(slen)
     if len(v) != slen:
       sys.stderr.write("extra field read ERROR\n")
       return False
   if not has_id or not obsslen == 2:
     sys.stderr.write("no proper extra header ERROR\n")
     return False
  return True