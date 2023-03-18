This is an EXERO Geophysics rapid processing algoirthm for ground magnetic and VLF data. I built this project in the winter of 2021. It was never complete, but is working. 

Please refer to update PDF for usage. No additional functionality was added after 2021-02-20 update.

[20210220_Update.pdf](https://github.com/celw10/EXERO/files/11008531/20210220_Update.pdf)

MAG/VLF Dictionary

lat - latitude </br>
lon - longitude </br>
elevation - GPS MSL </br>
nT - Magnetometer Total Field </br>
sq - Signal Quality 99 -best to 00 - no signal </br>
cor-nT - Corrected field if correction made via base station </br>
sat - Number of GPS satellites </br>
Position-type -  GPS status ie S for single </br>
Time - GPS time (note unit may have UTC offset) </br>
picket-x - if unit is configured can have a number when button pressed, default is * </br>
picket-y if unit is configured can have a number when button pressed, default is * </br>
slope - optional used defined value to indicate if on slope (not used for calculation) </br>
kHz - Station frequency </br>
ip - In-phase component </br>
op - Out-of-phase component </br>
h1- signal pickup on h1 coil </br>
h2 - signal pick-up on h2 coil </br>
pT - signal strength </br>
