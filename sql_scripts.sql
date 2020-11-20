-- load data
create schema kaggle;
alter table fire set schema kaggle;
create schema fdr;
create schema bc;
create schema plots;
create schema main;
ALTER TABLE kaggle.fire ADD PRIMARY KEY (objectid);
CREATE INDEX objectid_idx ON kaggle.fire(objectid);
CREATE INDEX stid_idx ON fdr.weather(st_id);

-- average fire weather data by county
create table plots.weather_county_year as
select avg(t1.bi) as avg_bi,avg(t1.wind) as avg_wind,avg(t1.tmp) as avg_tmp,t1.year, t2.geom from
fdr.weather t1, county t2
where ST_Contains(t2.geom, t1.geom)
group by t2.gid, t1.year;
-- average fire weather data by state
create table plots.weather_state_year as
select avg(t1.bi) as avg_bi,avg(t1.wind) as avg_wind,avg(t1.tmp) as avg_tmp,t1.year, t2.geom from
state t2, fdr.weather t1
where ST_Contains(t2.geom, t1.geom)
group by t2.gid, t1.year;

-- join the fire data and weather data by county
create table main.county_join as
select t1.objectid,t1.fire_year, t1.stat_cause_code, t1.stat_cause_descr,
t1.fire_size, t1.fire_size_class, t1.latitude, t1.longitude,
t1.state as fire_state, t1.county, t1.start_date, t1.end_date, t1.geom as fire_geom, t2.name,
t3.*
from kaggle.fire t1
left join county t2
on ST_Contains(t2.geom, t1.geom)
left join fdr.weather t3
on t1.start_date=t3.date and ST_Contains(t2.geom, t3.geom);
-- combine county and name to one column
-- add one column of distance between the fire and the station within the same county
create table main.county_join_up as
select *,
case when county is not null then county
else name
end fire_county,
case when ST_Distance(fire_geom, geom)=0 then 1
else 1/ST_Distance(fire_geom, geom) end dist_to_st
from main.county_join;

alter table main.county_join_up
drop column county,
drop column name,
drop column st_name,
drop column date,
drop column year,
drop column state;

-- calculated weighted weather information for each wildfire
create table main.fire_weather_county as
select objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
latitude, longitude, fire_state, start_date, end_date, fire_geom,
sum(tmp*weight) as tmp, sum(rh*weight) as rh, sum(wind*weight) as wind, sum(erc*weight) as erc,
sum(bi*weight) as bi, sum(sc*weight) as sc, sum(kbdi*weight) as kbdi
from
(select *, dist_to_st/sum_dist as weight from
(select *,
sum(dist_to_st) over (partition by objectid) as sum_dist
from main.county_join_up) t) t1
group by objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
latitude, longitude, fire_state, start_date, end_date, fire_geom;

-- join the fire data and weather data by state
create table main.state_join as
select t1.*, t3.st_id, t3.tmp as tmp_s, t3.rh as rh_s, t3.wind as wind_s,
t3.erc as erc_s, t3.bi as bi_s, t3.sc as sc_s, t3.kbdi as kbdi_s, t3.geom from
main.fire_weather_county t1
left join state t2
on t1.fire_state=t2.stusps
left join fdr.weather t3
on t1.start_date=t3.date and t2.name=t3.state
where fire_year>=2005;
-- add one column of distance between the fire and the station within the same county
create table main.state_join_up as
select *,
case when ST_Distance(fire_geom, geom)=0 then 1
else 1/ST_Distance(fire_geom, geom) end dist_to_st
from main.state_join;

-- calculated weighted weather information for each wildfire
create table main.fire_weather_state as
select objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
latitude, longitude, fire_state, start_date, end_date, fire_geom, tmp, rh, wind, erc, bi,
sc, kbdi,
sum(tmp_s*weight) as tmp_s, sum(rh_s*weight) as rh_s, sum(wind_s*weight) as wind_s,
sum(erc_s*weight) as erc_s, sum(bi_s*weight) as bi_s, sum(sc_s*weight) as sc_s,
sum(kbdi_s*weight) as kbdi_s from
(select *, dist_to_st/sum_dist as weight from
(select *,
sum(dist_to_st) over (partition by objectid) as sum_dist
from main.state_join_up) t) t1
group by objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
latitude, longitude, fire_state, start_date, end_date, fire_geom, tmp, rh, wind, erc, bi,
sc, kbdi;

-- fill the null value from county join with state join data
create table main.fire_weather as
select objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
latitude, longitude, fire_state, start_date, end_date, fire_geom,
coalesce(tmp,tmp_s) as tmp, coalesce(rh,rh_s) as rh, coalesce(wind,wind_s) as wind,
coalesce(erc,erc_s) as erc, coalesce(bi,bi_s) as bi, coalesce(sc,sc_s) as sc,
coalesce(kbdi,kbdi_s) as kbdi
from main.fire_weather_state;

-- create table for analysis (with only states we are interested in)
create table main.analysis as
select *, end_date-start_date as cont_time from main.fire_weather
where fire_state in
('CA','WA','OR','AZ','NV','ID','UT','WY','MT','NM','CO','FL','GA','SC','NC');

update main.analysis
set cont_time=null
where cont_time<0;
