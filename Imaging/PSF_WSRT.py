import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import track_simulator

# Let's draw the WSRT antenna's configuration in ENU coordinates
# Coordinates are defined in "westbork_92m_constants.py" We copy-past the code here (easier)
#*************************************************************************************
#WESTERBORK 92m ANTENNA COORDINATES
#*************************************************************************************

NO_ANTENNA = 14
NO_BASELINES = NO_ANTENNA * (NO_ANTENNA - 1) / 2 + NO_ANTENNA
global CENTRE_CHANNEL
CENTRE_CHANNEL = 299792458 / 1e9 #Wavelength of 1 GHz
#Antenna positions (from Measurement Set "ANTENNA" table)
#Here we assumed these are in Earth Centred Earth Fixed coordinates, see:
#https://en.wikipedia.org/wiki/ECEF
#http://casa.nrao.edu/Memos/229.html#SECTION00063000000000000000
ANTENNA_POSITIONS = np.array([[ 3828763.10544699,   442449.10566454,  5064923.00777   ],
                              [ 3828746.54957258,   442592.13950824,  5064923.00792   ],
                              [ 3828729.99081359,   442735.17696417,  5064923.00829   ],
                              [ 3828713.43109885,   442878.2118934 ,  5064923.00436   ],
                              [ 3828696.86994428,   443021.24917264,  5064923.00397   ],
                              [ 3828680.31391933,   443164.28596862,  5064923.00035   ],
                              [ 3828663.75159173,   443307.32138056,  5064923.00204   ],
                              [ 3828647.19342757,   443450.35604638,  5064923.0023    ],
                              [ 3828630.63486201,   443593.39226634,  5064922.99755   ],
                              [ 3828614.07606798,   443736.42941621,  5064923.        ],
                              [ 3828603.04244797,   443831.78969855,  5064922.99868   ],
                              [ 3828594.76228709,   443903.3070022 ,  5064922.99963   ],
                              [ 3828454.02400919,   445119.11903552,  5064922.99071   ],
                              [ 3828445.7469865 ,   445190.63592735,  5064922.98793   ]])
ARRAY_LATITUDE = 52.9157 #Equator->North
ARRAY_LONGITUDE = 6.5950 #Greenwitch->East, prime -> local meridian
REF_ANTENNA = 0
#Conversion from ECEF -> ENU:
#http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
slambda = np.sin(np.deg2rad(ARRAY_LONGITUDE))
clambda = np.cos(np.deg2rad(ARRAY_LONGITUDE))
sphi = np.sin(ARRAY_LONGITUDE)
cphi = np.cos(ARRAY_LATITUDE)
ecef_to_enu = [[-slambda,clambda,0],
               [-clambda*sphi,-slambda*sphi,cphi],
               [clambda*cphi,slambda*cphi,sphi]]
ENU = np.empty(ANTENNA_POSITIONS.shape)
for a in range(0,NO_ANTENNA):
    ENU[a,:] = np.dot(ecef_to_enu,ANTENNA_POSITIONS[a,:])
ENU -= ENU[REF_ANTENNA]


plt.figure(figsize=(8,4))
plt.title("ENU coordinate antenna - WSRT")
plt.scatter(ENU[:,0],
            ENU[:,1],
            c=ENU[:,2],
            marker="+",
            cmap="winter")
plt.colorbar()
plt.xlabel("E (m)")
plt.ylabel("N (m)")
plt.xlim([-1000, 3000])
plt.ylim([-30, 30])

# Plot uv coverage
uv_5min = track_simulator.sim_uv_v2(0.0, 90.0, 5/60, 60/3600, ENU, ARRAY_LATITUDE, True, True, CENTRE_CHANNEL)/CENTRE_CHANNEL/1e4
uv_6h_90_dec = track_simulator.sim_uv_v2(0.0, 90.0, 6, 60/3600, ENU, ARRAY_LATITUDE, True, True, CENTRE_CHANNEL)/CENTRE_CHANNEL/1e4
uv_6h_60_dec = track_simulator.sim_uv_v2(0.0, 60.0, 6, 60/3600, ENU, ARRAY_LATITUDE, True, True, CENTRE_CHANNEL)/CENTRE_CHANNEL/1e4
uv_6h_45_dec = track_simulator.sim_uv_v2(0.0, 45.0, 6, 60/3600, ENU, ARRAY_LATITUDE, True, True, CENTRE_CHANNEL)/CENTRE_CHANNEL/1e4
uv_6h_20_dec = track_simulator.sim_uv_v2(0.0, 20.0, 6, 60/3600, ENU, ARRAY_LATITUDE, True, True, CENTRE_CHANNEL)/CENTRE_CHANNEL/1e4
uv_6h_0_dec = track_simulator.sim_uv_v2(0.0, 0.0, 6, 60/3600, ENU, ARRAY_LATITUDE, True, True, CENTRE_CHANNEL)/CENTRE_CHANNEL/1e4



plt.show()

