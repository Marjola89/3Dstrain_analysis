###################################################
######### 3D strain plotting ######################
########### Marjola Thanaj ########################
###################################################
rm(list = ls(all = TRUE)) 


base.dir<- "~/cardiac/Experiments_of_Maria/3Dstrain_analysis"
setwd(base.dir)
l <- substr(list.dirs(recursive=F),3,11)

#### Prepare the middle atlas for all subjects ####

nPoints <- c(1:29146)
Phases<-c(1:50)

Data_rr<-matrix(0,ncol=50, nrow = length(nPoints))
Data_cc<-matrix(0,ncol=50, nrow = length(nPoints))
Data_ll<-matrix(0,ncol=50, nrow = length(nPoints))

ErrDataFrame<-vector(mode = "list",length=10)
EllDataFrame<-vector(mode = "list",length=10)
EccDataFrame<-vector(mode = "list",length=10)

for (i in 1:10){
  dir<- paste("~/cardiac/Experiments_of_Maria/3Dstrain_analysis/",l[i],"/middle_atlas", sep="")#
  
  for (iF in 1:50){
    Dat <- (fread(paste(dir,"/neopheno_", iF,".txt", sep=""), header=T))
    Data_ll[,iF]<-Dat$ELZZ
    Data_rr[,iF]<-Dat$ELRR
    Data_cc[,iF]<-Dat$ELTT
  }
  ErrDataFrame[[i]]<-Data_rr
  EllDataFrame[[i]]<-Data_ll
  EccDataFrame[[i]]<-Data_cc
}
Er<-matrix(0,nrow=10, ncol=50)
El<-matrix(0,nrow=10, ncol=50)
Ec<-matrix(0,nrow=10, ncol=50)

for (iE in 1:10){
  Er[iE,]<-as.data.frame(ksmooth(Phases,colMeans(ErrDataFrame[[iE]], na.rm=TRUE), "normal", bandwidth = 5, n.points = 50), type="l")$y
  El[iE,]<-as.data.frame(ksmooth(Phases,colMeans(EllDataFrame[[iE]], na.rm=TRUE), "normal", bandwidth = 5, n.points = 50), type="l")$y
  Ec[iE,]<-as.data.frame(ksmooth(Phases,colMeans(EccDataFrame[[iE]], na.rm=TRUE), "normal", bandwidth = 5, n.points = 50), type="l")$y
}

#
#### Prepare the tag atlas for all subjects ####

nPoints <- c(1:50656)
Phases<-c(1:50)

ErrDataFrame<-vector(mode = "list",length=10)
EllDataFrame<-vector(mode = "list",length=10)
EccDataFrame<-vector(mode = "list",length=10)

for (i in 1:10){
  dir<- paste("~/cardiac/Experiments_of_Maria/3Dstrain_analysis/",l[i],"/tag_atlas", sep="")#
  
  Data_rr<-fread(paste(dir,"/Sradial.txt", sep=""), header=F)
  Data_cc<-fread(paste(dir,"/Scirc.txt", sep=""), header=F)
  Data_ll<-fread(paste(dir,"/Slong.txt", sep=""), header=F)
  
  ErrDataFrame[[i]]<-Data_rr
  EllDataFrame[[i]]<-Data_ll
  EccDataFrame[[i]]<-Data_cc
}

Er<-matrix(0,nrow=10, ncol=50)
El<-matrix(0,nrow=10, ncol=50)
Ec<-matrix(0,nrow=10, ncol=50)

for (iE in 1:10){
  Er[iE,]<-as.data.frame(ksmooth(Phases,colMeans(ErrDataFrame[[iE]], na.rm=TRUE), "normal", bandwidth = 5, n.points = 50), type="l")$y
  El[iE,]<-as.data.frame(ksmooth(Phases,colMeans(EllDataFrame[[iE]], na.rm=TRUE), "normal", bandwidth = 5, n.points = 50), type="l")$y
  Ec[iE,]<-as.data.frame(ksmooth(Phases,colMeans(EccDataFrame[[iE]], na.rm=TRUE), "normal", bandwidth = 5, n.points = 50), type="l")$y
}

#
#### Plot #####
library(ggplot2)
library(ggpubr)
library(reshape2)
melt_plotdata<-melt(El, na.rm = TRUE)
colnames(melt_plotdata)<-c("No","Phases","Ell")

# my.formula <- y ~ x 
my.formula <- y ~ s(x, bs = "cs")
p1  <- ggplot(melt_plotdata, aes(x=Phases,y=Ell))+
  geom_point() + labs(x = "Phases", y = "Ell ") + 
  geom_point(alpha=0.06, colour="steelblue2", pch=19) +
  stat_density2d(geom="density2d", aes(alpha=..level..), colour="magenta3", size=1.5, contour=TRUE, show.legend = F) +
  theme(panel.border = element_blank(),
        axis.line.y = element_line(colour="black"),
        axis.text = element_text(colour="black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x  = element_text(size=16),
        axis.text.y  = element_text(size=16),
        axis.ticks.x = element_line(),
        axis.title.x  = element_text(size=18, vjust=0.3, face="bold"),
        axis.title.y  = element_text(size=18, face = "bold", vjust=0.9, angle = 90),
        axis.line = element_line(size = 1.2, linetype = "solid"),
        axis.ticks = element_line(size = 1), legend.position="none"
  ) +
  geom_smooth(method = "gam", formula = my.formula, colour="black", size=1.2, level=0.99)

p1
