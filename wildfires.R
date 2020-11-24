#November 23, 2020
#This script loads wildfire data for analysis

#Required packages
library(RSQLite)
library(haven)
library(sp)
library(rgdal)
library(data.table)
library(factoextra)
library(ggplot2)

#Import data (stored in data subfolder of parent folder)
#Wildfire data
con <- dbConnect(drv=RSQLite::SQLite(), 
                 dbname="./../Project Data/FPA_FOD_20170508.sqlite")
fires <- dbReadTable(con, "Fires")
#Fire weather data
fire_weather <- haven::read_dta("./../Project Data/fdr.dta")
#Fire weather zone shapefiles
fwz <- rgdal::readOGR(dsn= "./../Project Data/fz03mr20", 
                      layer="fz03mr20", verbose=FALSE)

#Get calendar dates from year and day of year
fires$DATE <- as.Date(fires$DISCOVERY_DOY - 1, 
                      origin=paste(fires$FIRE_YEAR, 1, 1, sep="-"))

#Get calendar dates: Stata date origin is Jan 1, 1960
fire_weather$Date <- as.Date(fire_weather$Date, origin="1960-01-01")

#Convert data to spatial points data frames, and make projections identical
sp::coordinates(fires) <- ~ LONGITUDE + LATITUDE
sp::coordinates(fire_weather) <- ~ Long + Lat
proj4string(fire_weather) <- proj4string(fires)
proj4string(fwz) <- proj4string(fires)

#Add fire weather zone each wildfire was in
fires$ZONE <- sp::over(fires, fwz)[, c("ZONE")]
fire_weather$Zone <- sp::over(fire_weather, fwz)[, c("ZONE")] #needlessly slow

#Aggregate fire weather data by zone
#data.table package performs aggregations quickly
fire_weather_dt <- data.table::as.data.table(fire_weather)
fire_weather_dt[, lapply(.SD, mean),  by = .(Zone, Date), 
                .SDcols = c("BI","ERC","Wind","KBDI")]
fire_weather_lt <- fire_weather_dt[, lapply(.SD, mean),  by = .(Zone), 
                  .SDcols = c("BI","ERC","Wind","KBDI")]
fire_weather_lt  <- na.omit(fire_weather_lt) #remove NAs

#K-means clustering
fw_sc <- fire_weather_lt
fw_sc$Zone <- NULL
fw_sc <- scale(fw_sc)
fw_sc <- as.data.frame(fw_sc)
set.seed(32936628)
factoextra::fviz_nbclust(fw_sc, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)
km.res <- kmeans(fw_sc, 3, nstart = 25)
km.res$centers
fire_weather_lt <- cbind(fire_weather_lt, cluster=km.res$cluster)
fire_weather_lt$ZONE <- fire_weather_lt$Zone
fire_weather_lt$Zone <- NULL
fwz_cl <- merge(x=fwz, y=fire_weather_lt, by = "ZONE")
fwz_cl <- subset(fwz_cl, LON > -140 & LON < -66)
fwz_cl@data$id <- rownames(fwz_cl@data)
fwz_ff <- fortify(fwz_cl, region="id")
fwz_ff_cl <- merge(fwz_ff, fwz_cl@data, by="id")
west_short <- c("CA","OR","WA","AZ","UT","NV","ID","NM","CO","WY","MT")
fwz_ff_cl <- subset(fwz_ff_cl, STATE %in% west_short)
fwz_ff_cl$cluster <- factor(as.character(fwz_ff_cl$cluster))
jpeg(file="Plots/cluster.jpeg")
ggplot() + geom_polygon(data = fwz_ff_cl, 
                        aes(x = long, y = lat, group = group, fill = cluster), 
                        colour = "black") +
  scale_fill_discrete(name="Cluster", 
                      labels=c("Dry and Windy","High Fire Risk",
                               "Low Fire Risk","Not in a Cluster"))
dev.off()
