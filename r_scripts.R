rm(list=ls())
library(RPostgres)
library(DBI)
library(RSQLite)
library(sp)
library(rpostgis)
library(tidyverse)
library(haven)
library(raster)
library(rgdal)

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


# land cover
land_raster<-raster("C:/Liwei/data_mining/project/data/land_cover/resampled.tif")
band<-as.data.frame(values(land_raster))
xy<-as.data.frame(xyFromCell(land_raster,1:ncell(land_raster)))
land_cover<-cbind(xy,band)
colnames(land_cover)=c('x','y','band')
land_cover<-land_cover[which(land_cover$band!=0),]
rownames(land_cover) <- 1:nrow(land_cover)
coords <- SpatialPoints(land_cover[, c("x", "y")])
spland <- SpatialPointsDataFrame(coords, land_cover)
pgInsert(con,name=c("public","land_cover"),data.obj=spland[1:500000,])
pgInsert(con,name=c("public","land_cover"),data.obj=spland[500001:1000000,])
pgInsert(con,name=c("public","land_cover"),data.obj=spland[1000001:1500000,])
pgInsert(con,name=c("public","land_cover"),data.obj=spland[1500001:2000000,])
pgInsert(con,name=c("public","land_cover"),data.obj=spland[2000001:2090545,])

# population density by county
pop<-read.csv("C:/Users/liuco/Downloads/co-est2019-alldata.csv")
names(pop) <- tolower(names(pop))
pop_census<-pop[,c('state','county','census2010pop')]
colnames(pop_census)<-c('state','county','population')
write.csv(pop_census, "C:/Liwei/data_mining/project/data/population_dens/pop.csv", row.names = FALSE)
# county area data (square miles)
area<-read.csv("C:/Users/liuco/Downloads/LND01.csv")
names(area) <- tolower(names(area))
area_2010<-area[2:nrow(area),c('stcou','lnd110210d')]
colnames(area_2010)<-c('code','area')
write.csv(area_2010, "C:/Liwei/data_mining/project/data/population_dens/area.csv", row.names = FALSE)