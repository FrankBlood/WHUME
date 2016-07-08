import os, sys

c_path = os.getcwd()
for f,subf,fns in os.walk(sys.argv[1]):
    for fn in fns:
        img_path = c_path+"/"+f+"/"+fn
        print img_path, f[f.rfind("/")+1:]