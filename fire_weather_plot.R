#Load fire weather data
library(haven)
fire_weather2016 <- read_dta("/Volumes/NO NAME/WFAS/fdr2016.dta")
#select fire weather data from a date near mid-year
july_fwd <- subset(fire_weather2016, Date == 20634)
#keep data within continental U.S.
july_fwd <- subset(july_fwd, longitude > -140 & longitude < -66)
july_fwd <- subset(july_fwd, !(State %in% c("Alaska","Hawaii","Puerto Rico","a")))
#select Western States
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
#BI can be replaced with ERC, KBDI, Wind, etc
#the plot takes some time to render on my computer
library(mapproj)
jpeg(file="fire_weather_plot.jpeg")
map + geom_point(data = july_fwd, aes(x = longitude, y = latitude, color = BI),
                 size = 1, alpha = 1) + 
  coord_map(projection = "albers", lat0 = 39, lat1 = 45) +
  labs(title = "US RAWS Stations", subtitle = "Mid 2016", size = "BI") + 
  theme(legend.position = "right") + 
  scale_color_gradient(low="blue", high="red")
dev.off()

"Below plots map of land cover"
#http://zevross.com/blog/2018/10/02/
#creating-beautiful-demographic-maps-in-r-with-the-tidycensus-and-tmap-packages/
#part-2-creating-beautiful-maps-with-tmap
#tmap not compatible with ggplot, though
library(sf)
library(stringr)
# Create shp file
shp <- st_read("/Users/gregoryklevans/Downloads/acs_2012_2016_county_us_B27001/acs_2012_2016_county_us_B27001.shp",
               stringsAsFactors = FALSE) %>%
  mutate(STFIPS = stringr::str_sub(GEOID, 1, 2))
shp <- st_transform(shp, CRS("+proj=utm +zone=18 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"))

library(tmap)

# Load the raster data
data(land)

# Remove AK and HI
library(dplyr)
shp_rmAKHI <- filter(shp, !STFIPS %in% c("02", "15"))
shp_west <- filter(shp, 
                   STFIPS %in% c("04","06","08","16","30","32","35","41","49","53","56"))

# Create a color palette for land cover
# This was taken directly from the tmap documentation
pal20 <- c("#003200", "#3C9600", "#006E00", "#556E19", "#00C800", 
           "#8CBE8C", "#467864", "#B4E664", "#9BC832", "#EBFF64", "#F06432", 
           "#9132E6", "#E664E6", "#9B82E6", "#B4FEF0", "#646464", "#C8C8C8", 
           "#FF0000", "#FFFFFF", "#5ADCDC")

# Map the raster data and assign the bounding box of
# the county layer. Add the county layer on top
jpeg(file="land_cover.jpeg")
tm_shape(land, bbox = shp_west) +
  tm_raster("cover", palette = pal20, alpha = 0.8) +
  tm_shape(shp_west) + 
  tm_borders(alpha = 0.4, col = "black") +
  tm_layout(inner.margins = c(0.06, 0.10, 0.10, 0.08)) +
  tm_layout(legend.position = c("left","bottom"))
dev.off()
