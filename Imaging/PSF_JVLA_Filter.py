import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import track_simulator
from scipy.interpolate import griddata
from convolutional_gridder import grid_ifft
from AA_filter import AA_filter


# Let's draw the WSRT antenna's configuration in ENU coordinates

#*************************************************************************************
#JVLA A-CONFIGURATION ANTENNA COORDINATES
#*************************************************************************************

NO_ANTENNA = 27
NO_BASELINES = NO_ANTENNA * (NO_ANTENNA - 1) / 2 + NO_ANTENNA
global CENTRE_CHANNEL
CENTRE_CHANNEL = 299792458 / 1e9 #Wavelength of 1 GHz
#Antenna positions (from Measurement Set "ANTENNA" table)
#Here we assumed these are in Earth Centred Earth Fixed coordinates, see:
#https://en.wikipedia.org/wiki/ECEF
#http://casa.nrao.edu/Memos/229.html#SECTION00063000000000000000
ANTENNA_POSITIONS = np.array([[-1601614.0612 , -5042001.67655,  3554652.4556 ],
                              [-1602592.82353, -5042055.01342,  3554140.65277],
                              [-1604008.70191, -5042135.83581,  3553403.66677],
                              [-1605808.59818, -5042230.07046,  3552459.16736],
                              [-1607962.41167, -5042338.15777,  3551324.88728],
                              [-1610451.98712, -5042471.38047,  3550021.01156],
                              [-1613255.37344, -5042613.05253,  3548545.86436],
                              [-1616361.55414, -5042770.44074,  3546911.38642],
                              [-1619757.27801, -5042937.57456,  3545120.33283],
                              [-1600801.8806 , -5042219.38668,  3554706.38228],
                              [-1599926.05941, -5042772.99258,  3554319.74284],
                              [-1598663.0464 , -5043581.42675,  3553766.97336],
                              [-1597053.09556, -5044604.74775,  3553058.94731],
                              [-1595124.91894, -5045829.51558,  3552210.61536],
                              [-1592894.06565, -5047229.19866,  3551221.18004],
                              [-1590380.58836, -5048810.32526,  3550108.40109],
                              [-1587600.20193, -5050575.97608,  3548885.37942],
                              [-1584460.89944, -5052385.73479,  3547599.95893],
                              [-1601147.88523, -5041733.85511,  3555235.91485],
                              [-1601061.91592, -5041175.90771,  3556057.98198],
                              [-1600929.96685, -5040316.40179,  3557330.27755],
                              [-1600780.99626, -5039347.46356,  3558761.48715],
                              [-1600592.69255, -5038121.38064,  3560574.80334],
                              [-1600374.8084 , -5036704.25301,  3562667.85595],
                              [-1600128.31399, -5035104.17725,  3565024.64505],
                              [-1599855.571  , -5033332.40332,  3567636.57859],
                              [-1599557.83837, -5031396.39194,  3570494.71676]])

ARRAY_LATITUDE = 34 + 4 / 60.0 + 43.497 / 3600.0 #Equator->North
ARRAY_LONGITUDE = -(107 + 37 / 60.0 + 03.819 / 3600.0) #Greenwitch->East, prime -> local meridian
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



DECLINATION = 30.0
RA = 0.0
fig = plt.figure(figsize=(16, 32))
ax=fig.add_subplot(321)
ax.set_title("JVLA A ENU ANTENNA COORDINATES")
ax.plot(ENU[:,0],
           ENU[:,1],
           'r+')
ax.set_xlabel("E (m)")
ax.set_ylabel("N (m)")
vla_a_uvw = track_simulator.sim_uv_v2(RA, DECLINATION, 0.5, 60/3600.0, ENU, ARRAY_LATITUDE)






#*************************************************************************************
#JVLA D-CONFIGURATION ANTENNA COORDINATES
#*************************************************************************************

NO_ANTENNA = 27
NO_BASELINES = NO_ANTENNA * (NO_ANTENNA - 1) / 2 + NO_ANTENNA
#Antenna positions (from Measurement Set "ANTENNA" table)
#Here we assumed these are in Earth Centred Earth Fixed coordinates, see:
#https://en.wikipedia.org/wiki/ECEF
#http://casa.nrao.edu/Memos/229.html#SECTION00063000000000000000
ANTENNA_POSITIONS = np.array([[-1601710.017000 , -5042006.925200 , 3554602.355600],
                              [-1601150.060300 , -5042000.619800 , 3554860.729400],
                              [-1600715.950800 , -5042273.187000 , 3554668.184500],
                              [-1601189.030140 , -5042000.493300 , 3554843.425700],
                              [-1601614.091000 , -5042001.652900 , 3554652.509300],
                              [-1601162.591000 , -5041828.999000 , 3555095.896400],
                              [-1601014.462000 , -5042086.252000 , 3554800.799800],
                              [-1601185.634945 , -5041978.156586 , 3554876.424700],
                              [-1600951.588000 , -5042125.911000 , 3554773.012300],
                              [-1601177.376760 , -5041925.073200 , 3554954.584100],
                              [-1601068.790300 , -5042051.910200 , 3554824.835300],
                              [-1600801.926000 , -5042219.366500 , 3554706.448200],
                              [-1601155.635800 , -5041783.843800 , 3555162.374100],
                              [-1601447.198000 , -5041992.502500 , 3554739.687600],
                              [-1601225.255200 , -5041980.383590 , 3554855.675000],
                              [-1601526.387300 , -5041996.840100 , 3554698.327400],
                              [-1601139.485100 , -5041679.036800 , 3555316.533200],
                              [-1601315.893000 , -5041985.320170 , 3554808.304600],
                              [-1601168.786100 , -5041869.054000 , 3555036.936000],
                              [-1601192.467800 , -5042022.856800 , 3554810.438800],
                              [-1601173.979400 , -5041902.657700 , 3554987.517500],
                              [-1600880.571400 , -5042170.388000 , 3554741.457400],
                              [-1601377.009500 , -5041988.665500 , 3554776.393400],
                              [-1601180.861480 , -5041947.453400 , 3554921.628700],
                              [-1601265.153600 , -5041982.533050 , 3554834.858400],
                              [-1601114.365500 , -5042023.151800 , 3554844.944000],
                              [-1601147.940400 , -5041733.837000 , 3555235.956000]]);
ARRAY_LATITUDE = 34 + 4 / 60.0 + 43.497 / 3600.0 #Equator->North
ARRAY_LONGITUDE = -(107 + 37 / 60.0 + 03.819 / 3600.0) #Greenwitch->East, prime -> local meridian
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

ax=fig.add_subplot(322)
ax.set_title("JVLA D ENU ANTENNA COORDINATES")
ax.plot(ENU[:,0],
        ENU[:,1],
        'r+')
ax.set_xlabel("E (m)")
ax.set_ylabel("N (m)")
vla_d_uvw = track_simulator.sim_uv_v2(RA, DECLINATION, 0.5, 60/3600.0, ENU, ARRAY_LATITUDE)



ax=fig.add_subplot(323)
ax.set_title("JVLA A uv-coverage")
ax.plot(vla_a_uvw[:,0]/CENTRE_CHANNEL,
           vla_a_uvw[:,1]/CENTRE_CHANNEL,
           'r+',markersize=1)
ax.plot(-vla_a_uvw[:,0]/CENTRE_CHANNEL,
           -vla_a_uvw[:,1]/CENTRE_CHANNEL,
           'b+',markersize=1)
ax.set_xlabel("u ($\lambda$)")
ax.set_ylabel("v ($\lambda$)")
ax=fig.add_subplot(324)
ax.set_title("JVLA D uv-coverage")
ax.plot(vla_d_uvw[:,0]/CENTRE_CHANNEL,
           vla_d_uvw[:,1]/CENTRE_CHANNEL,
           'r+',markersize=1)
ax.plot(-vla_d_uvw[:,0]/CENTRE_CHANNEL,
           -vla_d_uvw[:,1]/CENTRE_CHANNEL,
           'b+',markersize=1)
ax.set_xlabel("u ($\lambda$)")
ax.set_ylabel("v ($\lambda$)")

# more about the next bit in the gridding section:
IM_SIZE_DEGREES = 0.1
scale = np.deg2rad(IM_SIZE_DEGREES)

tabulated_filter = AA_filter(3,63,"sinc")
dirty_sky_vla_a, psf_vla_a = grid_ifft(np.ones([vla_a_uvw.shape[0],1,1], dtype=np.complex64),
                                       vla_a_uvw[:,0:2] * scale,
                                       np.array([CENTRE_CHANNEL]),
                                       2048,2048,
                                       tabulated_filter)
dirty_sky_vla_d, psf_vla_d = grid_ifft(np.ones([vla_d_uvw.shape[0],1,1], dtype=np.complex64),
                                       vla_d_uvw[:,0:2] * scale,
                                       np.array([CENTRE_CHANNEL]),
                                       2048,2048,
                                       tabulated_filter)
ax=fig.add_subplot(325)
ax.set_title("JVLA A PSF")
ax.imshow(np.real(psf_vla_a),
          cmap="cubehelix",
          extent=[RA - IM_SIZE_DEGREES/2,
                  RA + IM_SIZE_DEGREES/2,
                  DECLINATION - IM_SIZE_DEGREES/2,
                  DECLINATION + IM_SIZE_DEGREES/2])
ax.set_xlabel("RA")
ax.set_ylabel("DEC")
ax=fig.add_subplot(326)
ax.set_title("JVLA D PSF")
ax.imshow(np.real(psf_vla_d),
          cmap="cubehelix",
          extent=[RA - IM_SIZE_DEGREES/2,
                  RA + IM_SIZE_DEGREES/2,
                  DECLINATION - IM_SIZE_DEGREES/2,
                  DECLINATION + IM_SIZE_DEGREES/2])
ax.set_xlabel("RA")
ax.set_ylabel("DEC")


plt.show()

