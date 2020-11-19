rm(list=ls())
library(RPostgres)
library(DBI)
library(RSQLite)
library(sp)
library(rpostgis)
library(tidyverse)
library(haven)

# load kaggle ata
con <- dbConnect(drv=RSQLite::SQLite(), "C:/Liwei/data_mining/project/FPA_FOD_20170508.sqlite")
data<-dbGetQuery(conn=con,statement = "select * from Fires")
head(data)
names(data) <- tolower(names(data))
data$start_date <- as.Date(data$discovery_doy - 1, 
                      origin=paste(data$fire_year, 1, 1, sep="-"))
data$end_date <- as.Date(data$cont_doy - 1, 
                           origin=paste(data$fire_year, 1, 1, sep="-"))
coords <- SpatialPoints(data[, c("longitude", "latitude")])
spdata <- SpatialPointsDataFrame(coords, data)
# upload data to pgadmin
drv<-dbDriver("PostgreSQL")
con<- dbConnect(RPostgres::Postgres(), dbname="postgres",host="localhost",port=5432,
                user="postgres",password="postgres")
pgInsert(con,name=c("kaggle","fire"),data.obj=spdata[1:600000,])
pgInsert(con,name=c("kaggle","fire"),data.obj=spdata[600001:1200000,])
pgInsert(con,name=c("kaggle","fire"),data.obj=spdata[1200001:1880465,])

# load fdr data
fdr2005<-read_dta("C:/Users/liuco/Downloads/fdr2005.dta")
fdr2006<-read_dta("C:/Users/liuco/Downloads/fdr2006.dta")
fdr2007<-read_dta("C:/Users/liuco/Downloads/fdr2007.dta")
fdr2008<-read_dta("C:/Users/liuco/Downloads/fdr2008.dta")
fdr2009<-read_dta("C:/Users/liuco/Downloads/fdr2009.dta")
fdr2010<-read_dta("C:/Users/liuco/Downloads/fdr2010.dta")
fdr2011<-read_dta("C:/Users/liuco/Downloads/fdr2011.dta")
fdr2012<-read_dta("C:/Users/liuco/Downloads/fdr2012.dta")
fdr2013<-read_dta("C:/Users/liuco/Downloads/fdr2013.dta")
fdr2014<-read_dta("C:/Users/liuco/Downloads/fdr2014.dta")
fdr2015<-read_dta("C:/Users/liuco/Downloads/fdr2015.dta")
fdr2016<-read_dta("C:/Users/liuco/Downloads/fdr2016.dta")
fdr2005$year<-2005
fdr2006$year<-2006
fdr2007$year<-2007
fdr2008$year<-2008
fdr2009$year<-2009
fdr2010$year<-2010
fdr2011$year<-2011
fdr2012$year<-2012
fdr2013$year<-2013
fdr2014$year<-2014
fdr2015$year<-2015
fdr2016$year<-2016
fdr2014<-fdr2014[,which(colnames(fdr2014)!='st_nameia')]
colnames(fdr2016)<-c("Long","Lat","st_id","st_name","Elev","Mdl",
                     "Tmp","RH","Wind","PPT","ERC","BI","SC",
                     "KBDI","HUN","THOU","TEN","STL","ADJ",
                     "IC","S1","S2","S3","S4","S5","State",
                     "Date","dtst","STL5","year")
fdr<-rbind(fdr2005, fdr2006, fdr2007, fdr2008, fdr2009, fdr2010,
           fdr2011, fdr2012, fdr2013, fdr2014, fdr2015, fdr2016)
names(fdr) <- tolower(names(fdr))
fdr$date<-as.Date(fdr$date, origin="1960-01-01")
summary(fdr)
# convert fdr data to spatial point data frame
coords <- SpatialPoints(fdr[, c("long", "lat")])
spfdr <- SpatialPointsDataFrame(coords, fdr)
# upload fdr data to pgadmin
pgInsert(con,name=c("fdr","weather"),data.obj=spfdr[1:1000000,])
pgInsert(con,name=c("fdr","weather"),data.obj=spfdr[1000001:2000000,])
pgInsert(con,name=c("fdr","weather"),data.obj=spfdr[2000001:3000000,])
pgInsert(con,name=c("fdr","weather"),data.obj=spfdr[3000001:4000000,])
pgInsert(con,name=c("fdr","weather"),data.obj=spfdr[4000001:4340400,])

# load basicincident data
bc2014 <- read.delim("C:/Users/liuco/Downloads/basicincident2014.txt",
                     header=TRUE, sep="^")
names(bc2014) <- tolower(names(bc2014))
bc2016 <- read.delim("C:/Users/liuco/Downloads/basicincident2016.txt",
                     header=TRUE, sep="^")
names(bc2016) <- tolower(names(bc2016))
bc2013 <- read.delim("C:/Users/liuco/Downloads/basicincident2013.txt",
                     header=TRUE, sep="^")
names(bc2013) <- tolower(names(bc2013))
bc2012 <- read.delim("C:/Users/liuco/Downloads/basicincident2012.txt",
                     header=TRUE, sep="^")
names(bc2012) <- tolower(names(bc2012))
bc2010 <- read.dbf("C:/Users/liuco/Downloads/basicincident2010.dbf")
names(bc2010) <- tolower(names(bc2010))
bc2011 <- read.dbf("C:/Users/liuco/Downloads/basicincident2011.dbf")
names(bc2011) <- tolower(names(bc2011))
bc2009 <- read.dbf("C:/Users/liuco/Downloads/basicincident2009.dbf")
names(bc2009) <- tolower(names(bc2009))
bc2008 <- read.dbf("C:/Users/liuco/Downloads/basicincident2008.dbf")
names(bc2008) <- tolower(names(bc2008))
bc2007 <- read.dbf("C:/Users/liuco/Downloads/basicincident2007.dbf")
names(bc2007) <- tolower(names(bc2007))

# upload to pgadmin
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2007[1:1000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2007[1000001:1500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2007[1500001:2197537,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2008[1:1000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2008[1000001:1500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2008[1500001:2178599,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2009[1:1000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2009[1000001:1500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2009[1500001:2072850,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2010[1:500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2010[500001:1000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2010[1000001:1500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2010[1500001:2000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2010[2000001:2221660,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2011[1:500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2011[500001:1000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2011[1000001:1500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2011[1500001:2000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2011[2000001:2311716,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2012[1:500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2012[500001:1000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2012[1000001:1500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2012[1500001:2000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2012[2000001:2120288,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2013[1:500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2013[500001:1000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2013[1000001:1500000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2013[1500001:2000000,])
pgInsert(con,name=c("bc","fire_report"),data.obj=bc2013[2000001:2003907,])
# change memory limit
memory.limit()
memory.limit(size = 16300)
# upload 2014 data
for (i in 0:45){
  pgInsert(con,name=c("bc","fire_report_2014"),data.obj=bc2014[(1+i*500000):500000*(i+1),])
}