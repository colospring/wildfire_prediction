#Import wildfire data
library(RSQLite)
con <- dbConnect(drv=RSQLite::SQLite(), 
                 dbname="/Users/gregoryklevans/Downloads/FPA_FOD_20170508.sqlite")
fires <- dbReadTable(con, "Fires")

#Get calendar dates
fires$DATE <- sapply(fires$FIRE_YEAR, toString)    #convert fire years to string
fires$DATE <- paste0(fires$DATE , "-01-01")      #set dates to beginning of year
fires$DATE <- as.Date(fires$DISCOVERY_DOY, origin=fires$DATE)   #add day of year

#Import fire weather data
library(haven)
fire_weather <- read_dta("/Volumes/NO NAME/WFAS/fdr.dta")
#Get calendar dates: Stata date origin is Jan 1, 1960
fire_weather$Date <- as.Date(fire_weather$Date, origin="1960-01-01")

#Convert data to spatial points data frames
library(sp)
coordinates(fires) <- ~ LONGITUDE + LATITUDE
coordinates(fire_weather) <- ~ Long + Lat

#Import fire weather zone shapefiles
library(rgdal)
fwz <- readOGR(dsn= "/Users/gregoryklevans/Downloads/fz03mr20", 
               layer="fz03mr20", verbose=FALSE)