This is an EXERO Geophysics rapid processing algoirthm for ground magnetic and VLF data.

I built this project in the winter of 2021. It was never complete, but is in working order. 

MAG/VLF DATA DICTIONARY

lat - latitude </br>
lon - longitude
elevation - GPS MSL
nT - Magnetometer Total Field
sq - Signal Quality 99 -best to 00 - no signal
cor-nT - Corrected field if correction made via base station
sat - Number of GPS satellites
Position-type -  GPS status ie S for single
Time - GPS time (note unit may have UTC offset)
picket-x - if unit is configured can have a number when button pressed, default is *
picket-y if unit is configured can have a number when button pressed, default is *
slope - optional used defined value to indicate if on slope (not used for calculation)
kHz - Station frequency
ip - In-phase component
op - Out-of-phase component
h1- signal pickup on h1 coil
h2 - signal pick-up on h2 coil
pT - signal strength
