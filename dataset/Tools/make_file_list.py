
from os import listdir
from os.path import isfile,isdir,join
mypath="/research/cvlshare/Databases/zhang835/FORD_MSU_png/"
onlydirs=[f for f in listdir(mypath) if isdir(join(mypath,f))]
onlydirs=sorted(onlydirs)

file_ = open('file_listing.txt','w')

for d in onlydirs:
	for f in listdir(join(mypath,d)):
		if isfile(join(mypath,d,f)):
			file_.write(join(mypath,d,f)+"\n")		

file_.close()

#print(onlydirs)
