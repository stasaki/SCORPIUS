options(stringsAsFactors = FALSE)

#install.packages("SCORPIUS")
library(SCORPIUS)
library(tidyverse)

# Setup python3.7 environment ####
# Required library pandas, numpy, sklearn


# Please specify file location ####
input_data = "data.txt"
outdir = "./out/"
python_loc = "/opt/anaconda3/bin/python" # depends on your enviroment
se_loc = "./functions/" # available at https://github.com/stasaki/SCORPIUS/tree/main/functions/
dir.create(outdir,recursive = T)

# SpectralEmbedding ####
n_pc=40
k=10
cmd = paste0(python_loc," ",se_loc,"SpectralEmbedding.py ",
             input_data," ",outdir," ",k," ",n_pc)
system(cmd)

# SCORPIUS ####
out = read.delim(paste0(outdir,"k",k,"_pc",n_pc,".gzip"),header = T,sep = "\t")
set.seed(1234)
traj <- infer_trajectory(out[,c("Component1","Component2")],k=5)
traj$path%>%
  as.data.frame()%>%
  mutate(time=traj$time)%>%
  mutate(SAMPLE.ID=out$SAMPLE.ID) -> ds
saveRDS(ds,file=paste0(outdir,"k",k,"_pc",n_pc,".rds"))
