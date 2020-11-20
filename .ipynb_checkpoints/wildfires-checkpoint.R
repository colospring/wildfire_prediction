#November 17, 2020
#This script loads wildfire data for analysis

#Required packages
library(RSQLite)
library(haven)
library(sp)
library(rgdal)
library(data.table)

#Import data (stored in data subfolder of parent folder)
#Wildfire data
con <- dbConnect(drv=RSQLite::SQLite(), dbname="./../Project Data/FPA_FOD_20170508.sqlite")
fires <- dbReadTable(con, "Fires")
fire_weather <- haven::read_dta("./../Project Data/fdr.dta") #Fire weather data
fwz <- rgdal::readOGR(dsn= "./../Project Data/fz03mr20", layer="fz03mr20", verbose=FALSE) #Fire weather zone shapefiles

#Get wildfire calendar dates from year and day of year
fires$DATE <- as.Date(fires$DISCOVERY_DOY - 1, origin=paste(fires$FIRE_YEAR, 1, 1, sep="-"))

#Get fire weather calendar dates: Stata date origin is Jan 1, 1960
fire_weather$Date <- as.Date(fire_weather$Date, origin="1960-01-01")

#Convert data to spatial points data frames, and make projections identical
sp::coordinates(fires) <- ~ LONGITUDE + LATITUDE
sp::coordinates(fire_weather) <- ~ Long + Lat
proj4string(fire_weather) <- proj4string(fires)
proj4string(fwz) <- proj4string(fires)

#Add fire weather zone each wildfire was in
fires$ZONE <- sp::over(fires, fwz)[, c("ZONE")]
fire_weather$Zone <- sp::over(fire_weather, fwz)[, c("ZONE")] #needlessly slow

#Aggregate fire weather data by zone: data.table package performs aggregations quickly
fire_weather_dt <- data.table::as.data.table(fire_weather)
fire_weather_dt[, lapply(.SD, mean),  by = .(Zone, Date), .SDcols = c("BI","ERC","Wind","KBDI")]
