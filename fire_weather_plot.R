#Load fire weather data
library(haven)
fire_weather2016 <- read_dta("/Volumes/NO NAME/WFAS/fdr2016.dta")
#select fire weather data from a date near mid-year
july_fwd <- subset(fire_weather2016, Date == 20634)
#keep data within continental U.S.
july_fwd <- subset(july_fwd, longitude > -140 & longitude < -66)
july_fwd <- subset(july_fwd, !(State %in% c("Alaska","Hawaii","Puerto Rico","a")))
#Can select only Western States, but plot looks stretched
west_long <- c("California","Oregon","Washington","Arizona","Utah","Nevada",
               "Idaho","New Mexico","Colorado","Wyoming","Montana")
july_fwd <- subset(july_fwd, State %in% west_long)

#load fire weather zone shapefiles
#source: https://www.weather.gov/gis/FireZones
library(rgdal)
fwz <- readOGR(dsn= "/Users/gregoryklevans/Downloads/fz03mr20", 
  layer="fz03mr20", verbose=FALSE)
fwz <- subset(fwz, LON > -140 & LON < -66)
fwz <- subset(fwz, !(STATE %in% c("GU","PR","VI","AS","AK","HI")))
west_short <- c("CA","OR","WA","AZ","UT","NV","ID","NM","CO","WY","MT")
fwz <- subset(fwz, STATE %in% west_short)

#create map of fire weather zones in continental U.S.
library(ggplot2)
map <- ggplot() + 
  geom_polygon(data = fwz, 
               aes(x = long, y = lat, group = group), 
               colour = "black", fill = NA)

#plot map along with Burning Index measurements for each RAWS station
#the plot takes some time to render on my computer
#proportions seem off, especially with Western States only
library(mapproj)
map + geom_point(data = july_fwd, aes(x = longitude, y = latitude, color = BI),
                 size = 1, alpha = 1) + 
  coord_map(projection = "albers", lat0 = 39, lat1 = 45) +
  labs(title = "US RAWS Stations", subtitle = "Mid 2016", size = "BI") + 
  theme(legend.position = "right") + 
  scale_color_gradient(low="blue", high="red")
