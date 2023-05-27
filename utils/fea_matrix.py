import mne
import numpy as np
import torch
import mne_connectivity as mne_c
from mne_connectivity import spectral_connectivity_time
from collections import OrderedDict
from mne.time_frequency import psd_array_welch

# 生成PLV联通矩阵
class ComputeGraph():
    def __init__(self,method,data):
        self.method = method
        self.data = data # (n_samplers, time_step, n_channels)
    def con_matrix(self,):
        for sampler in range(len(self.data)):
            data1 = self.data[sampler,:,:] # ()
            data1 = np.expand_dims(data1, axis=0)
            data1 = np.swapaxes(data1,1,2) # (epoch, n_channels, time_step)
            n_channels = data1.shape[1]
            conn_matrix = np.zeros((n_channels,n_channels))
            for ch_idx in range(n_channels):
                spec_conn4 = spectral_connectivity_time(data=data1,
                                                        method=self.method,
                                                        n_cycles=3,
                                                        sfreq=128,
                                                        average=True,
                                                        # indices= (np.array([0,0,0,0]),np.array([0,1,2,3])),1
                                                        indices= ([ch_idx]*n_channels, np.arange(0,n_channels)),
                                                        fmin = 1.0,
                                                        fmax = 40.0,
                                                        # fmin=(1.0, 4.0, 7.5, 13.0, 16.0, 30.0), 
                                                        # fmax=(4.0, 7.5, 13.0, 16.0, 30.0, 40.0),
                                                        freqs = np.linspace(1, 40, 10),
                                                        faverage=True
                                                        )
                conn_matrix[:,ch_idx] = spec_conn4.get_data().flatten()
            print(conn_matrix) # 对称的矩阵
            return conn_matrix
x = np.load('data/data_s12.npy')
x = np.swapaxes(x,1,2)
x32 = np.load('data/data_norm.npy')
# Graph = ComputeGraph(method='plv',data=x)
# Graph.con_matrix()

# psds, freqs = psd_array_welch(x=x32[1,:,:],sfreq=128,fmin=0,fmax=40,average='mean',n_per_seg=128)
# print(psds)

class ComputeFeature(object):
    """
    parameter:
    ---------
    data.shape = (n_samplers, n_channels, time_step)
    method: string 'coh' 'plv'
    return:
    ---------
    feature_matrix.shape = (n_samplers, n_channels, n_features)    
    con_matrix.shape = (n_smaplers, n_channels, n_channels)
    """
    def __init__(self,data,method):
        self.data = data 
        self.method = method
    def compute_psd(self,):
        n_samplers,n_channels,_ = self.data.shape
        feature_matrix = np.zeros((n_samplers,n_channels,6)) 
        con_matrix = np.zeros((n_samplers,n_channels,n_channels))
        for i in range(n_samplers):
            sampler_data = self.data[i,:,:]
            psds, freqs = psd_array_welch(x=sampler_data,sfreq=128,fmax=50.0,)
            psd_dB = 10 * np.log10(psds) # Convert power to dB scale.
            band_powers = self.get_bands_waves_power(psd_welch=psd_dB,freqs=freqs) # (n_channels, n_bands)=(12, 6)
            feature_matrix[i,:,:] = band_powers
            # 计算邻接矩阵
            for ch_idx in range(n_channels):
                data1 = np.expand_dims(sampler_data,axis=0)
                spec_conn = spectral_connectivity_time(data=data1,
                                                        method=self.method,
                                                        n_cycles=3,
                                                        sfreq=128,
                                                        average=True,
                                                        indices= ([ch_idx]*n_channels, np.arange(0,n_channels)),
                                                        fmin = 1.0,
                                                        fmax = 40.0,
                                                        freqs = np.linspace(1, 40, 10),
                                                        faverage=True
                                                        )
                con_matrix[i,:,ch_idx] = spec_conn.get_data().flatten()
        return feature_matrix,con_matrix # feature_matrix.shape = (n_samplers, n_channels, n_features)    con_matrix.shape = (n_smaplers, n_channels, n_channels)
    # 计算bands的psd
    def get_bands_waves_power(self, psd_welch, freqs):
        brain_waves = OrderedDict({
            "delta" : [1.0, 4.0],
            "theta": [4.0, 7.5],
            "alpha": [7.5, 13.0],
            "lower_beta": [13.0, 16.0],
            "higher_beta": [16.0, 30.0],
            "gamma": [30.0, 40.0]
	    })
        band_powers = np.zeros((psd_welch.shape[0], 6)) # (n_channels, n_bands)
        for wave_idx,wave in enumerate(brain_waves.keys()):
            if wave_idx == 0:
                band_freqs_idx = np.argwhere((freqs <= brain_waves[wave][1])) # 结束频率 即返回 小于等于 fmax的所有频率的索引
            else:
                band_freqs_idx = np.argwhere((freqs >= brain_waves[wave][0])&(freqs <= brain_waves[wave][1])) # 即返回 大于等于fmin 小于等于fmax的所有频率的索引
            
            band_psd = psd_welch[:,band_freqs_idx.ravel()] # 若 band_freqs_idx.ravel() = [1, 2, 3, 4, 7] 则band_psd为返回所有行 第1, 2, 3, 4, 7列组成的新数组
            total_band_power = np.sum(band_psd, axis=1) # 将每列都相加 即得到在这个频段上所有的psd
            band_powers[:,wave_idx] = total_band_power
        return band_powers
compute_fea = ComputeFeature(x,'plv')
fea_m,conn = compute_fea.compute_psd()
print(fea_m.shape)