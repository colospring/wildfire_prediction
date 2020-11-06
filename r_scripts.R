rm(list=ls())
library(RPostgres)
library(DBI)
library(RSQLite)
library(sp)
library(rpostgis)

con <- dbConnect(drv=RSQLite::SQLite(), "C:/Liwei/data_mining/project/FPA_FOD_20170508.sqlite")
data<-dbGetQuery(conn=con,statement = "select * from Fires")
drv<-dbDriver("PostgreSQL")
con<- dbConnect(RPostgres::Postgres(), dbname="postgres",host="localhost",port=5432,
                user="postgres",password="postgres")
pgInsert(con,name=c("kaggle","fire"),data.obj=data)