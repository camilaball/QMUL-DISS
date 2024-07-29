if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("methylKit")

library("methylKit")

file.list <- list("/data/scratch/bt.../.cov.gz")

treatment1=rep(0,71)


myobj <- methRead(file.list,
                  sample.id=list("SRR4048940","SRR.."),
                  pipeline = "bismarkCoverage",
                  assembly="hg38",
                  treatment=treatment1,
                  mincov = 10,
                  context= "CpG",
                  resolution = "base")


myobj.filt <- filterByCoverage(myobj,
                               lo.count=10,
                               lo.perc=NULL,
                               hi.count=NULL,
                               hi.perc=99.9)

#Normalization
myobj.filt.norm <- normalizeCoverage(myobj.filt, method="median")

#Merge Data
meth <- unite(myobj.filt.norm, destrand=FALSE,mc.cores=2,min.per.group=3L)#need to add minimum per group


#final file to csv
write.csv(meth,"meth.csv")
