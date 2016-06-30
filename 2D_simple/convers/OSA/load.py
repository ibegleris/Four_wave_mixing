from __future__ import division
from scipy.io import loadmat,savemat
import matplotlib.pylab as plt
from peakdetect import peakdetect
import os
import numpy as np
import sys
from scipy.io import savemat
from peakdetect import peakdetect
sys.path.insert(1, '../..')
def dbm2w(dBm):
    """This function converts a power given in dBm to a power given in W.
       Inputs::
           dBm(float): power in units of dBm
       Returns::
           Power in units of W (float)
    """
    return 1e-3*10**((dBm)/10.)


def w2dbm(W):
    """This function converts a power given in W to a power given in dBm.
       Inputs::
           W(float): power in units of W
       Returns::
           Power in units of dBm(float)
    """
    return 10.*np.log10(W)+30





files = [f for f in os.listdir('.') if os.path.isfile(f)]
for  i, f in enumerate(files):
	if f[-2::] == 'py':
		files = np.delete(files, i)
		break
files.sort()

new_files = []
for f in files:
	if f[4:7] == 'and':
		if f[8] !='.':
			lam_str = f[0:4]+'.'+f[7:9]+'.mat'
		else:
			lam_str = f[0:4]+'.'+f[7]+'.mat'
	else:
		lam_str = f
	print f, 'becomes', lam_str
	new_files.append(lam_str)
	mat = loadmat(f)
	savemat('new_mats/'+lam_str,mat)

lam_large = np.zeros([4,len(new_files),60])

D_large = np.zeros([4,len(new_files),60])
plt.close('all')
for i,files in enumerate(['1553.25.mat']):

	if files[0] != '1':
		break
	print i
	mat = loadmat('new_mats/'+files)
	D = mat['D_01']
	print np.shape(D)
	lam = mat['lam_01']

	D_11 = w2dbm(dbm2w(mat['D_11a']) + dbm2w(mat['D_11b']))
	lam_11 = mat['lam_11a']
	#plt.plot(lam,D)
	#plt.plot(lam_11,D_11)
	#plt.show()
	whereis = np.where(D_11 == np.max(D_11))[1][0]
	D_11 = np.delete(D_11,whereis,1)
	D = np.delete(D,whereis,1)
	lam = np.delete(lam, whereis,1)
	lam_11 = np.delete(lam_11, whereis,1)
	print(np.max(D))
	print(np.max(D_11))
	#D_11 -=np.max(D_11)
	#D -= np.max(D)
	print(np.max(D_11))
	plt.close('all')
	plt.plot(lam,D)
	plt.plot(lam_11,D_11)
	plt.ylim(-60,0)
	plt.xlim(1541,1557)
	plt.grid()
	plt.show()
	sys.exit()
	for j in range(np.shape(D)[1]):
		
		#finds the lam of the pumps for the splitting of the search for the LP01
		peaks = peakdetect(D[:,j],lam[:,j],8,0.7)
		peaks = np.asanyarray(peaks[0])
		lam_peaks = peaks[:][:,0]
		D_peaks = peaks[:][:,1]
		clutter_01 = np.where(D_peaks<-59)
		D_peaks = np.delete(D_peaks, clutter_01)
		lam_peaks = np.delete(lam_peaks, clutter_01)

		peaks_11 = peakdetect(D_11[:,j],lam_11[:,j],8,0.7)
		peaks_11 = np.asanyarray(peaks_11[0])
		lam_peaks_11 = peaks_11[:][:,0]
		D_peaks_11 = peaks_11[:][:,1]
		
		clutter_11 = np.where(D_peaks_11<-59)
		lam_peaks_11 = np.delete(lam_peaks_11, clutter_11)
		D_peaks_11 = np.delete(D_peaks_11, clutter_11)
		

		lam_pump1 = np.min(lam_peaks[np.argsort(D_peaks)[-2::]])
		lam_pump2 = np.max(lam_peaks_11[np.argsort(D_peaks_11)[-2::]])


		#for the degenerate scattering
		pos2 = np.where(lam_peaks < lam_pump1)[0] # find the possitions of the peaks with a smaller wavelength than lamp1
		pos_deg = np.where(D_peaks[pos2] == np.max(D_peaks[pos2]))
		D_deg = D_peaks[pos2[pos_deg]]
		lam_deg = lam_peaks[pos2[pos_deg]]



		#for the signal 
		#pos3 = np.where((lam_peaks > lam_pump1) & (lam_peaks < lam_pump2))[0] # find the possitions of the peaks with a smaller wavelength than lamp1
		#pos_sig = np.where(D_peaks[pos3] == np.max(D_peaks[pos3]))
		#D_sig = D_peaks[pos3[pos_sig]]
		#lam_sig = lam_peaks[pos3[pos_sig]]
		
		#for the phase conjugation 
		#pos4 = np.where((lam_peaks_11 > lam_pump1) & (lam_peaks_11 < lam_pump2))[0]
		#pos_pc = -1#np.where(D_peaks_11[pos4] == np.max(D_peaks_11[pos4]))
		#D_pc = D_peaks_11[pos4[pos_pc]]
		#lam_pc = lam_peaks_11[pos4[pos_pc]]

		#for the brag scattering
		#pos1 = np.where(lam_peaks_11 >lam_pump2)[0] # find the possitions of the peaks with a larger wavelength than lamp2
		#pos_bs = np.where(D_peaks_11[pos1] == np.max(D_peaks_11[pos1]))
		#D_bs = D_peaks_11[pos1[pos_bs]]
		#lam_bs = lam_peaks_11[pos1[pos_bs]]
		print(D_deg)
		print(lam_deg)

		lam_large[0,i,j] =  lam_deg[0]#, 0,0,0#lam_sig, lam_bs,lam_pc
		D_large[0,i,j] =  D_deg[0]#, 0,0,0#D_sig, D_bs, D_pc
	plt.plot(lam_11,D_11)
	plt.plot(lam,D)
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
	D_save,lam_save = np.zeros([4,np.shape(D)[1]]), np.zeros([4,np.shape(D)[1]])
	for ii in range(1):
		lam_plot = np.delete(lam_large[ii,i,:], np.where(lam_large[ii,i,:]== 0.0))
		D_plot = np.delete(D_large[ii,i,:], np.where(D_large[ii,i,:]== 0.0))
		#plt.plot(lam_plot,D_plot,'o', label = ii)
		D_save[ii,:] = D_plot
		lam_save[ii,:] = lam_plot
	#plt.plot(lam_large[:,i,:],D_large[:,i,:],'o')
	plt.legend()
	plt.show()
	#sys.exit()
	#sys.exit()

saver = {}

saver['D'] = D_save
saver['lam'] = lam_save
savemat('mod_inst.mat', saver)
#sys.exit()