-- load data
create schema kaggle;
alter table fire set schema kaggle;
create schema fdr;
create schema bc;
create schema plots;
create schema main;
ALTER TABLE kaggle.fire ADD PRIMARY KEY (objectid);
-- create indexes
CREATE INDEX objectid_idx ON kaggle.fire(objectid);
CREATE INDEX stid_idx ON fdr.weather(st_id);
CREATE INDEX sp_idx ON land_cover USING GIST (geom);
CREATE INDEX sp_fire_idx ON main.analysis USING GIST (fire_geom);

-- add state abbreviation to fire weather data
create table fdr.weather_up as
select t1.*, t2.stusps from fdr.weather t1
left join state t2
on ST_Contains(t2.geom, t1.geom);

create table fdr.weather_upp as
select t1.*,
case when t1.stusps is not null then t1.stusps
else t2.stusps end state_abbr
from fdr.weather_up t1
left join state t2
on t1.state=t2.name;

drop table fdr.weather;
drop table fdr.weather_up;

create table fdr.weather as
select * from fdr.weather_upp
where state_abbr is not null;

drop table fdr.weather_upp;
alter table fdr.weather drop column state, drop column stusps;
alter table fdr.weather rename column state_abbr to state;

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

-- join the fire data and weather data by fire zone (delete null values)
create table main.fz_join as
select t1.objectid,t1.fire_year, t1.stat_cause_code, t1.stat_cause_descr,
t1.fire_size, t1.fire_size_class, t1.latitude, t1.longitude,
t1.state as fire_state, t1.county, t1.start_date, t1.end_date, t1.geom as fire_geom,
t3.*
from kaggle.fire t1
join fire_zone t2
on t1.state=t2.state and ST_Contains(t2.geom, t1.geom)
join fdr.weather t3
on t1.state=t3.state and t1.start_date=t3.date and ST_Contains(t2.geom, t3.geom)
where fire_year>=2005 and t1.state in
('CA','WA','OR','AZ','NV','ID','UT','WY','MT','NM','CO','FL','GA','SC','NC')

-- add one column of distance between the fire and the station within the same fire zone
create table main.fz_join_up as
select *,
case when ST_Distance(fire_geom, geom)=0 then 1
else 1/ST_Distance(fire_geom, geom) end dist_to_st
from main.fz_join;

-- calculated weighted weather information by distance for each wildfire
create table main.fire_weather_fz as
select objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
latitude, longitude, fire_state, start_date, end_date, fire_geom,
sum(tmp*weight) as tmp, sum(rh*weight) as rh, sum(wind*weight) as wind, sum(erc*weight) as erc,
sum(bi*weight) as bi, sum(sc*weight) as sc, sum(kbdi*weight) as kbdi
from
(select *, dist_to_st/sum_dist as weight from
(select *,
sum(dist_to_st) over (partition by objectid) as sum_dist
from main.fz_join_up) t) t1
group by objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
latitude, longitude, fire_state, start_date, end_date, fire_geom;

-- -- join the fire data and weather data by state
-- create table main.state_join as
-- select t1.*, t3.st_id, t3.tmp as tmp_s, t3.rh as rh_s, t3.wind as wind_s,
-- t3.erc as erc_s, t3.bi as bi_s, t3.sc as sc_s, t3.kbdi as kbdi_s, t3.geom from
-- main.fire_weather_fz t1
-- left join state t2
-- on t1.fire_state=t2.stusps
-- left join fdr.weather t3
-- on t1.start_date=t3.date and t2.name=t3.state
-- where fire_year>=2005;
-- -- add one column of distance between the fire and the station within the same county
-- create table main.state_join_up as
-- select *,
-- case when ST_Distance(fire_geom, geom)=0 then 1
-- else 1/ST_Distance(fire_geom, geom) end dist_to_st
-- from main.state_join;
--
-- -- calculated weighted weather information for each wildfire
-- create table main.fire_weather_state as
-- select objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
-- latitude, longitude, fire_state, start_date, end_date, fire_geom, tmp, rh, wind, erc, bi,
-- sc, kbdi,
-- sum(tmp_s*weight) as tmp_s, sum(rh_s*weight) as rh_s, sum(wind_s*weight) as wind_s,
-- sum(erc_s*weight) as erc_s, sum(bi_s*weight) as bi_s, sum(sc_s*weight) as sc_s,
-- sum(kbdi_s*weight) as kbdi_s from
-- (select *, dist_to_st/sum_dist as weight from
-- (select *,
-- sum(dist_to_st) over (partition by objectid) as sum_dist
-- from main.state_join_up) t) t1
-- group by objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
-- latitude, longitude, fire_state, start_date, end_date, fire_geom, tmp, rh, wind, erc, bi,
-- sc, kbdi;
--
-- -- fill the null value from county join with state join data
-- create table main.fire_weather as
-- select objectid, fire_year, stat_cause_code, stat_cause_descr, fire_size, fire_size_class,
-- latitude, longitude, fire_state, start_date, end_date, fire_geom,
-- coalesce(tmp,tmp_s) as tmp, coalesce(rh,rh_s) as rh, coalesce(wind,wind_s) as wind,
-- coalesce(erc,erc_s) as erc, coalesce(bi,bi_s) as bi, coalesce(sc,sc_s) as sc,
-- coalesce(kbdi,kbdi_s) as kbdi
-- from main.fire_weather_state;

-- create table for analysis (with only states we are interested in)
create table main.analysis as
select *,
case when end_date-start_date<0 then null
else end_date-start_date end cont_time
from main.fire_weather_fz;

-- create population table
create table pop_county(
	state int,
	county int NOT NULL,
	population real
);
-- merge geom to population table
create table pop_county_up as
select t2.*,t1.geom from county t1
join pop_county t2
on CAST(t1.countyfp AS int)=t2.county
and CAST(t1.statefp AS int)=t2.state;

drop table pop_county;
alter table pop_county_up rename to pop_county;
-- create a table of county areas
create table area_county(
	code int NOT NULL,
	area real
);
-- calculate population density by county
create table pop_county_dens as
select t1.*, t2.area, t1.population/t2.area as pop_dens from pop_county t1
join
(select right(cast(code as varchar(50)),3)::int as county,
left(right('0'|| cast(code as varchar(50)),5),2)::int as state, area
from area_county) t2
on t1.county=t2.county and t1.state=t2.state;

CREATE INDEX sp_pop_idx ON pop_county_dens USING GIST (geom);

-- merge population density to the main analysis table
create table main.analysis_up as
select t1.*, t2.pop_dens from main.analysis t1
join pop_county_dens t2
on ST_Contains(t2.geom, t1.fire_geom);

drop table main.analysis;
alter table main.analysis_up rename to analysis;

-- create land_class table
create table land_class(
	band int primary key,
	land_type VARCHAR(250),
	class VARCHAR(250)
);
-- add land_cover information to the analysis table
create table main.analysis_up as
select t1.*, t3.class as land_type from main.analysis t1
join land_cover_shp t2
on ST_Contains(t2.geom, t1.fire_geom)
join land_class t3
on t2.suitable=t3.band;

drop table main.analysis;
alter table main.analysis_up rename to analysis;
