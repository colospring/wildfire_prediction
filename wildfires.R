#Import wildfire data
library(RSQLite)
con <- dbConnect(drv=RSQLite::SQLite(), 
                 dbname="/Users/gregoryklevans/Downloads/FPA_FOD_20170508.sqlite")
fires <- dbReadTable(con, 'Fires')

#Get calendar dates
years <- sapply(fires$FIRE_YEAR, toString)
jan1 <- replicate(length(fires$FIRE_YEAR), "-01-01")
fires$DATE <- paste0(years, jan1)
fires$DATE <- as.Date(fires$DISCOVERY_DOY, origin = fires$DATE)
