import numpy as np
from math import exp, log, sqrt, radians,sin,cos
import numba
import matplotlib.pyplot as plt
import pyfftw

class Nanostructure:
    def __init__(self, settings_file="./settings.txt"):

        self._read_file(settings_file)

        self.CoordinateX = (np.tile(np.arange(self.NumberOfXPoints), (self.NumberOfYPoints,1)))*self.Resolution
        self.CoordinateY = (np.tile((np.arange(self.NumberOfYPoints)).reshape(-1,1), (1,self.NumberOfXPoints)))*self.Resolution
        self.DeltaH = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints))
        ## for height
        DefectXCoord = self.DefectInfo[:,1]/self.Resolution
        DefectYCoord = self.DefectInfo[:,2]/self.Resolution
        DefectRadiusCoord = self.DefectInfo[:,0]/self.Resolution
        DefectRadiusCoordSquare = DefectRadiusCoord**2
        self.Height = np.ndarray((self.NumberOfYPoints,self.NumberOfXPoints))
        random_int = np.random.randint(5,1004)
        
        for y in range(self.NumberOfYPoints):
            for x in range(self.NumberOfXPoints):
                random_int = np.random.randint(random_int+np.random.randint(999))
                self.Height[y][x] = random_int//100*2e-9
                if(self.HowManyDefects != 0):
                    for i in range(self.HowManyDefects):
                        a=DefectRadiusCoordSquare[i]-((x-DefectXCoord[i])**2 +(y-DefectYCoord[i])**2)
                        if(a > 0 ):
                            self.Height[y][x] = sqrt(a)*self.Resolution

        self.IncidentI = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints))
        self.IncidentEx = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints), dtype=np.complex)
        self.IncidentEy = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints), dtype=np.complex)
        self.ScatteredEx = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints), dtype=np.complex)
        self.ScatteredEy = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints), dtype=np.complex)
        self.TotalEx = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints), dtype=np.complex)
        self.TotalEy = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints), dtype=np.complex)
        self.TotalIntensity = np.zeros((self.NumberOfYPoints,self.NumberOfXPoints))

        self.ElementarySquare = self.Resolution**2

        self._set_Constants()

        self.H_max = self.Height.max()



    def _read_file(self, settings_file):

        file = open(settings_file,'r')

        self.NumberOfCycles      = int(file.readline().split(':')[1])
        self.NumberOfXPoints     = int(file.readline().split(':')[1])
        self.NumberOfYPoints     = int(file.readline().split(':')[1])
        self.Resolution          = int(file.readline().split(':')[1])*1E-9
        self.NumberOfTheFile     = int(file.readline().split(':')[1])
        self.ScanningStepX       = int(file.readline().split(':')[1])*1E-9 
        self.ScanningStepY       = int(file.readline().split(':')[1])*1E-9 
        self.PeriodOfStructure   = int(file.readline().split(':')[1])*1E-9
        self.PolarizationAngle   = int(file.readline().split(':')[1])
        self.PhaseShiftXY        = int(file.readline().split(':')[1])-90
        self.PhaseShift          = int(file.readline().split(':')[1])
        self.CenterOfBeamX       = int(file.readline().split(':')[1])*1E-9 
        self.CenterOfBeamY       = int(file.readline().split(':')[1])*1E-9
        self.MaxIncidentI        = float(file.readline().split(':')[1])
        self.BeamWidth           = int(file.readline().split(':')[1])*1e-6
        self.FieldKoefficient    = float(file.readline().split(':')[1])
        self.PPFieldKoefficient  = float(file.readline().split(':')[1])
        self.PPDecayL            = int(file.readline().split(':')[1])*1e-6
        # self.RandDefects         = bool(file.readline().split(':')[1])   ########## implement rand defects
        self.HowManyDefects      = int(file.readline().split(':')[1]) ## implement more than one defect
        self.DefectInfo =       np.ndarray((self.HowManyDefects,3))

        for i in range(1):
            # Rand_Radius = bool(file.readline().split(':')[1])
            Rand_Radius = False
            if(Rand_Radius):
                Radius = int(file.readline().split(':')[1])*1e-9
                RadOverRes = Radius/self.Resolution
                Radius = np.random.randint(2,RadOverRes+1)*self.Resolution    ## implement random radius
            else:
                Radius = int(file.readline().split(':')[1])*1e-9
            self.DefectInfo[i][0]=Radius
            # RandomDefect = file.readline().split(':')[1]
            RandomDefect = "False"
            if(RandomDefect in ["True", "true"]):
                X_pos = int(file.readline().split(':')[1])*1E-9
                Y_pos = int(file.readline().split(':')[1])*1E-9
                RadOverRes=Radius/self.Resolution
                self.DefectInfo[i][1]=np.random.randint(RadOverRes, self.NumberOfXPoints-RadOverRes+1)*self.Resolution
                self.DefectInfo[i][2]=np.random.randint(RadOverRes, self.NumberOfYPoints-RadOverRes+1)*self.Resolution
            else:
                
                OnCenter = file.readline().split(':')[1] in ["True\n", "true\n"]
                X_pos = int(file.readline().split(':')[1])*1E-9
                Y_pos = int(file.readline().split(':')[1])*1E-9
                if(OnCenter):
                    self.DefectInfo[i][1]=self.NumberOfXPoints*self.Resolution/2
                    self.DefectInfo[i][2]=self.NumberOfYPoints*self.Resolution/2
                else:
                    self.DefectInfo[i][1]=X_pos
                    self.DefectInfo[i][2]=Y_pos

    def _set_Constants(self):

        # to calculate number of Ti atoms
        self.MolarMass=47.88             #g/moll
        self.AvogadroNumber=6.022E23     #moll-^1
        self.HeatCapacityForSolid=0.52   #J/g*K
        self.HeatCapacityForLiquid=0.72  #J/g*K
        self.FusionHeatCapacity=385      # J/g
        self.EvaporationHeatCapacity=8970   # J/g
        self.DeltaTFromRoomTempToMelting=1655  # K
        self.DeltaTFromMeltingToEvaporation=1587 #K
        self.EvaporatedVolume=0.
        self.EvaporatedMass=0.
        self.AmoundOfTi=0.
        self.NumberOfTiAtoms=0.
        self.HeatCapacityForSolidHeating=self.HeatCapacityForSolid*self.DeltaTFromRoomTempToMelting #J/g
        self.HeatCapacityForLiquidHeating=self.HeatCapacityForLiquid*self.DeltaTFromMeltingToEvaporation #J/g
        self.TotalCapacity=self.HeatCapacityForSolidHeating+self.FusionHeatCapacity+self.HeatCapacityForLiquidHeating+self.EvaporationHeatCapacity #J/g
        self.TPuls=170E-15  #s Pulse duration
        self.SkinDept=30E-9 #m
        self.MinimalDeptForEvaporation=0.000001E-9 #m
        self.MassDensity=4500000 #g/m^3
        self.Ith=(self.TotalCapacity*self.SkinDept*self.MassDensity)/(self.TPuls) #the intensity for evaporation DeptForEvaporation (in fact at low Dept it is threshold intensity)
        self.DeptForEvaporation=0.
        self.CriticalHeight=50E-9      #m

        # to calculate number of O2 molecules

        self.MolarValue=22.4E-3         #m^3/moll
        self.AverageHeightOfAir=100E-9  #Assumed height of air where reaction is possible
        self.AmountOfMolecules=1*self.ElementarySquare*self.AverageHeightOfAir*self.AvogadroNumber/self.MolarValue  #astimated number of O2 molecules in the volume of 100 nm above ellementar square


        # setting TiO2 volume coefficient

        MolarMass_TiO2=79.8 #g/moll
        self.TiO2VolumeCoefficient=100*MolarMass_TiO2/(self.AvogadroNumber*self.MassDensity)

        

    def _NumberOfO2Molecules(self):

        NumberOfO2Molecules=self.AmountOfMolecules*np.exp(self.Height/-self.CriticalHeight)

        return NumberOfO2Molecules


    def _NumberOfTiAtoms(self):
        
        NumberOfTiAtoms = np.empty(self.TotalIntensity.shape)
        NumberOfTiAtoms[self.TotalIntensity>self.Ith] = (np.log(self.Ith/self.TotalIntensity[self.TotalIntensity>self.Ith]))*(-self.SkinDept * self.ElementarySquare * self.MassDensity * self.AvogadroNumber / self.MolarMass)
        NumberOfTiAtoms[self.TotalIntensity<=self.Ith] = -63 + 3.23e-4 * np.exp(self.TotalIntensity[self.TotalIntensity<=self.Ith]/(5.74e+14))
        np.clip(NumberOfTiAtoms, a_min=0. , a_max=None, out = NumberOfTiAtoms)

        return NumberOfTiAtoms



    def Execute(self, from_file = False):
       
        if(from_file):

            # 1st option

            self.CoordinateX = np.load("Surface_Data/self.CoordinateX.npy")
            self.CoordinateY = np.load("Surface_Data/self.CoordinateY.npy")
            self.DeltaH = np.load("Surface_Data/self.DeltaH.npy")
            self.Height = np.load("Surface_Data/self.Height.npy")

            #2nd option

            #   import pickle
            with open('surface_data.pkl', 'rb') as input:
                Array = pickle.load(input)

        self.IncidentI=self.MaxIncidentI*np.exp(-(np.square(self.CoordinateX-self.CenterOfBeamX)+np.square(self.CoordinateY-self.CenterOfBeamY))/(self.BeamWidth**2)) #Definition of initial field intensity
        Inc_sqr = np.sqrt(self.IncidentI)
        self.IncidentEx.real=Inc_sqr*round(cos(radians(self.PolarizationAngle)),10)*round(sin(radians(self.PhaseShiftXY)),10)
        self.IncidentEx.imag=Inc_sqr*round(cos(radians(self.PolarizationAngle)),10)*round(cos(radians(self.PhaseShiftXY)),10)
        self.IncidentEy.real=Inc_sqr*round(sin(radians(self.PolarizationAngle)),10)*round(cos(radians(self.PhaseShiftXY)),10)
        self.IncidentEy.imag=Inc_sqr*round(sin(radians(self.PolarizationAngle)),10)*round(sin(radians(self.PhaseShiftXY)),10) 
        
        for i in range(self.NumberOfCycles):

            self.IncidentI=self.MaxIncidentI*np.exp(-(np.square(self.CoordinateX-self.CenterOfBeamX)+np.square(self.CoordinateY-self.CenterOfBeamY))/(self.BeamWidth**2)) #Definition of initial field intensity
            Inc_sqr = np.sqrt(self.IncidentI)
            self.IncidentEx.real=Inc_sqr*round(cos(radians(self.PolarizationAngle)),10)*round(sin(radians(self.PhaseShiftXY)),10)
            self.IncidentEx.imag=Inc_sqr*round(cos(radians(self.PolarizationAngle)),10)*round(cos(radians(self.PhaseShiftXY)),10)          
            self.IncidentEy.real=Inc_sqr*round(sin(radians(self.PolarizationAngle)),10)*round(cos(radians(self.PhaseShiftXY)),10)
            self.IncidentEy.imag=Inc_sqr*round(sin(radians(self.PolarizationAngle)),10)*round(sin(radians(self.PhaseShiftXY)),10)                                                                                                                    #since we are considerind the face difference we always take EyReal=... and Eyimg=0
            self.TotalIntensity = self.IncidentI

            IOfIncIntens=np.sum(self.IncidentI)*self.ElementarySquare          #Integral of incident intensity


            self._all_calculation_fft()

            IOfResInt=np.sum(self.TotalIntensity)*self.ElementarySquare          #Integral of result intensity


            Temp=IOfIncIntens/IOfResInt                                                                   #Calculation of Koefficient for normalization of result intensity

            np.multiply(self.TotalIntensity, IOfIncIntens/IOfResInt, out=self.TotalIntensity)

            NTi = self._NumberOfTiAtoms()
            NO2 = self._NumberOfO2Molecules()
            Tio2_over_Esq= self.TiO2VolumeCoefficient/self.ElementarySquare

            self.DeltaH[NTi<NO2] =NTi[NTi<NO2]*Tio2_over_Esq
            self.DeltaH[NTi>=NO2] =NO2[NTi>=NO2]*Tio2_over_Esq

            self.Height+=self.DeltaH

            # save stuff 
            # save only the last not checked
            # save only the last checked and it is the last cycle

            # drawpic

            self.ScatteredEx.fill(0)
            self.ScatteredEy.fill(0)
            self.TotalEx.fill(0)
            self.TotalEy.fill(0)
            self.TotalIntensity.fill(0)


            self.CenterOfBeamX+=self.ScanningStepX
            self.CenterOfBeamY+=self.ScanningStepY



    def _all_calculation_fft(self):

        k=2*3.1415/self.PeriodOfStructure
        @numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
        def abs2(x):
            return x.real**2 + x.imag**2


        # self.TotalEx.fill(0)
        # self.TotalEy.fill(0)
        # self.ScatteredEx.fill(0)
        # self.ScatteredEy.fill(0)
        # self.TotalIntensity.fill(0)

        ## for_x_coords = np.tile(np.arange(self.NumberOfXPoints)-self.NumberOfXPoints//2, (self.NumberOfYPoints,1))
        ## for_y_coords = np.tile((np.arange(self.NumberOfYPoints)-self.NumberOfYPoints//2).reshape(-1,1), (1,self.NumberOfXPoints))
        dist_from_cent_x = (np.tile(np.arange(self.NumberOfXPoints)-self.NumberOfXPoints//2, (self.NumberOfYPoints,1)))*self.Resolution
        #dist_from_cent_x[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 1 # remove if
        dist_from_cent_x_sqr = np.square(dist_from_cent_x)
        dist_from_cent_y = (np.tile((np.arange(self.NumberOfYPoints)-self.NumberOfYPoints//2).reshape(-1,1), (1,self.NumberOfXPoints)))*self.Resolution
        #dist_from_cent_y[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 1 # remove if
        dist_from_cent_y_sqr = np.square(dist_from_cent_y)
         

        numberOfPoints = self.NumberOfXPoints * self.NumberOfYPoints
        Array_I = complex(0,1) # Array[1].I

        #####################################################################################################
        #calculating Exxscattered

        F = self.FieldKoefficient * self.Height * self.IncidentEx
        F[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if
        Radius = dist_from_cent_x_sqr + dist_from_cent_y_sqr
        Radius[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 1 # remove if
        PhaseRetarder = np.exp(Array_I * k * np.sqrt(Radius)) / np.sqrt(Radius)             

        PhoverRad = PhaseRetarder/ Radius                                                       # # saving it since it is used everywhere 

        G = PhoverRad * dist_from_cent_y_sqr
        G[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if
        #G = PhaseRetarder * dist_from_cent_y_sqr / Radius

        fft_f_x = np.fft.fft2(F)                  

        C = np.empty((self.NumberOfYPoints, self.NumberOfXPoints), dtype=np.complex)

        inverse_fourier = pyfftw.FFTW(fft_f_x * np.fft.fft2(G), C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEx += C / (numberOfPoints)      


        #####################################################################################################
        #calculating Exyscattered


        F = self.FieldKoefficient * self.Height * self.IncidentEy              
        F[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if                 
        #G = PhaseRetarder * dist_from_cent_x * dist_from_cent_y / Radius
        G =  PhoverRad * dist_from_cent_x * dist_from_cent_y 
        G[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if
        fft_f_y = np.fft.fft2(F)
        fft_g =  np.fft.fft2(G)


        inverse_fourier = pyfftw.FFTW(fft_f_y * fft_g, C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEx += C / (numberOfPoints) 
        


        #####################################################################################################
        #calculating Eyxscattered

        #G = PhaseRetarder * dist_from_cent_x* dist_from_cent_y  / Radius                       # G doesnt change

        inverse_fourier = pyfftw.FFTW(fft_f_x * fft_g, C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEy += C / (numberOfPoints) 



        #####################################################################################################
        #calculating Eyyscattered

        # G = PhaseRetarder * dist_from_cent_x_sqr  / Radius
        G = PhoverRad * dist_from_cent_x_sqr 
        G[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if

        inverse_fourier = pyfftw.FFTW(fft_f_x * np.fft.fft2(G), C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEy += C / (numberOfPoints) 


        #####################################################################################################
        #####################################################################################################

        #fft shift
        for x in np.arange(self.NumberOfXPoints//2 , self.NumberOfXPoints):
            self.ScatteredEy[:,[x,x-(self.NumberOfXPoints//2)]] = self.ScatteredEy[:,[x-(self.NumberOfXPoints//2),x]]
            self.ScatteredEx[:,[x,x-(self.NumberOfXPoints//2)]] = self.ScatteredEx[:,[x-(self.NumberOfXPoints//2),x]]

        self.ScatteredEx = np.flip(self.ScatteredEx.reshape(2,numberOfPoints//2), 0).reshape(self.NumberOfYPoints, self.NumberOfXPoints)
        self.ScatteredEy = np.flip(self.ScatteredEy.reshape(2,numberOfPoints//2), 0).reshape(self.NumberOfYPoints, self.NumberOfXPoints)

        # calculating total intensity
        self.TotalEx = self.ScatteredEx + self.IncidentEx
        self.TotalEy = self.ScatteredEy + self.IncidentEy
        self.TotalIntensity +=  abs2(self.TotalEx) + abs2(self.TotalEy)

        self.TotalEx.fill(0)
        self.TotalEy.fill(0)
        self.ScatteredEx.fill(0)
        self.ScatteredEy.fill(0)
        #####################################################################################################
        #####################################################################################################



        #####################################################################################################
        #calculating ExxPP

        F = self.PPFieldKoefficient * self.Height * self.IncidentEx
        F[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if
        # Radius = dist_from_cent_x_sqr + dist_from_cent_y_sqr                                  ## it doesnt change
        PhaseRetarder = np.exp((Array_I * k) - (1/self.PPDecayL) * np.sqrt(Radius))             
        PhoverRad = PhaseRetarder/ Radius                                                       # # saving it since it is used everywhere 
        G = PhoverRad * dist_from_cent_x_sqr
        G[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if

        fft_f_x = np.fft.fft2(F)

        inverse_fourier = pyfftw.FFTW(fft_f_x * np.fft.fft2(G), C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEx += C / (numberOfPoints) 

        #####################################################################################################
        #calculating Exyscattered

        F = self.PPFieldKoefficient * self.Height * self.IncidentEy
        F[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if
        G = PhoverRad * dist_from_cent_x * dist_from_cent_y
        G[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if
        fft_f_y = np.fft.fft2(F)
        fft_g =  np.fft.fft2(G)

        inverse_fourier = pyfftw.FFTW(fft_f_y * fft_g, C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEx += C / (numberOfPoints) 

        #####################################################################################################
        #calculating Eyxscattered
        inverse_fourier = pyfftw.FFTW(fft_f_x * fft_g, C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEy += C / (numberOfPoints) 

        #####################################################################################################
        #calculating Eyyscattered

        G = PhoverRad * dist_from_cent_y_sqr 
        G[self.NumberOfXPoints//2][self.NumberOfYPoints//2] = 0 # remove if
        inverse_fourier = pyfftw.FFTW(fft_f_y * np.fft.fft2(G), C, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD',axes=(-2, -1))
        inverse_fourier.execute()

        self.ScatteredEy += C / (numberOfPoints) 

        #####################################################################################################
        #####################################################################################################
        # fft shift
        for x in np.arange(self.NumberOfXPoints//2 , self.NumberOfXPoints):
            self.ScatteredEy[:,[x,x-(self.NumberOfXPoints//2)]] = self.ScatteredEy[:,[x-(self.NumberOfXPoints//2),x]]
            self.ScatteredEx[:,[x,x-(self.NumberOfXPoints//2)]] = self.ScatteredEx[:,[x-(self.NumberOfXPoints//2),x]]

        self.ScatteredEx = np.flip(self.ScatteredEx.reshape(2,numberOfPoints//2), 0).reshape(self.NumberOfYPoints, self.NumberOfXPoints)
        self.ScatteredEy = np.flip(self.ScatteredEy.reshape(2,numberOfPoints//2), 0).reshape(self.NumberOfYPoints, self.NumberOfXPoints)


        self.TotalEx = self.ScatteredEx + self.IncidentEx
        self.TotalEy = self.ScatteredEy + self.IncidentEy
        self.TotalIntensity +=  abs2(self.TotalEx) + abs2(self.TotalEy)

        #####################################################################################################
        #####################################################################################################

    def show_Structure(self):

        plt.imshow(self.Height, vmin = 0, vmax = self.H_max, cmap='gray')
        
        plt.show()



