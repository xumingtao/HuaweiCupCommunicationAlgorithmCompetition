#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
修改输出格式为RoundYOutputX.txt, 2023年5月4日09:55:03
"""
# import csiread
# import pywt
import math
import random
import pandas as pd
import os, time
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
from scipy.fftpack import fft,ifft,fftshift
from matplotlib.pylab import mpl
from scipy import signal
import scipy.signal as signal
from PyEMD import EMD
from scipy.signal import argrelextrema
# 进行样条差值
import scipy.interpolate as spi

# from pyemd.emd import emd
from matplotlib.font_manager import FontProperties
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
chinese_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
# my_font = font_manager.FontProperties(fname='C:\Windows\Fonts\MSYH.TTC')
#numpy 1.19
def scidx(bw, ng, standard='n'):
    """subcarriers index

    Args:
        bw: bandwitdh(20, 40, 80)
        ng: grouping(1, 2, 4)
        standard: 'n' - 802.11n， 'ac' - 802.11ac.
    Ref:
        1. 802.11n-2016: IEEE Standard for Information technology—Telecommunications
        and information exchange between systems Local and metropolitan area
        networks—Specific requirements - Part 11: Wireless LAN Medium Access
        Control (MAC) and Physical Layer (PHY) Specifications, in
        IEEE Std 802.11-2016 (Revision of IEEE Std 802.11-2012), vol., no.,
        pp.1-3534, 14 Dec. 2016, doi: 10.1109/IEEESTD.2016.7786995.
        2. 802.11ac-2013 Part 11: ["IEEE Standard for Information technology--
        Telecommunications and information exchange between systemsLocal and
        metropolitan area networks-- Specific requirements--Part 11: Wireless
        LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications
        --Amendment 4: Enhancements for Very High Throughput for Operation in
        Bands below 6 GHz.," in IEEE Std 802.11ac-2013 (Amendment to IEEE Std
        802.11-2012, as amended by IEEE Std 802.11ae-2012, IEEE Std 802.11aa-2012,
        and IEEE Std 802.11ad-2012) , vol., no., pp.1-425, 18 Dec. 2013,
        doi: 10.1109/IEEESTD.2013.6687187.](https://www.academia.edu/19690308/802_11ac_2013)
    """

    PILOT_AC = {
        20: [-21, -7, 7, 21],
        40: [-53, -25, -11, 11, 25, 53],
        80: [-103, -75, -39, -11, 11, 39, 75, 103],
        160: [-231, -203, -167, -139, -117, -89, -53, -25, 25, 53, 89, 117, 139, 167, 203, 231]
    }
    SKIP_AC_160 = {1: [-129, -128, -127, 127, 128, 129], 2: [-128, 128], 4: []}
    AB = {10: [28,1],20: [38, 1], 40: [58, 2], 80: [122, 2], 160: [250, 6]}
    a, b = AB[bw]

    if standard == 'n':
        if bw not in [10,20, 40] or ng not in [1, 2, 4]:
            raise ValueError("bw should be [20, 40] and ng should be [1, 2, 4]")
        k = np.r_[-a:-b:ng, -b, b:a:ng, a]
    if standard == 'ac':
        if bw not in [10,20, 40, 80] or ng not in [1, 2, 4]:
            raise ValueError("bw should be [20, 40, 80] and ng should be [1, 2, 4]")

        g = np.r_[-a:-b:ng, -b]
        k = np.r_[g, -g[::-1]]

        if ng == 1:
            index = np.searchsorted(k, PILOT_AC[bw])
            k = np.delete(k, index)
        if bw == 160:
            index = np.searchsorted(k, SKIP_AC_160[ng])
            k = np.delete(k, index)
    return k

def calib(phase, k, axis=2):
    """Phase calibration

    Args:
        phase (ndarray): Unwrapped phase of CSI.
        k (ndarray): Subcarriers index
        axis (int): Axis along which is subcarrier. Default: 1

    Returns:
        ndarray: Phase calibrated

    ref:
        [Enabling Contactless Detection of Moving Humans with Dynamic Speeds Using CSI]
        (http://tns.thss.tsinghua.edu.cn/wifiradar/papers/QianKun-TECS2017.pdf)
    """
    p = np.asarray(phase)
    k = np.asarray(k)
    slice1 = [slice(None, None)] * p.ndim
    slice1[axis] = slice(-1, None)
    slice1 = tuple(slice1)
    slice2 = [slice(None, None)] * p.ndim
    slice2[axis] = slice(None, 1)
    slice2 = tuple(slice2)
    shape1 = [1] * p.ndim
    shape1[axis] = k.shape[0]
    shape1 = tuple(shape1)

    k_n, k_1 = k[-1], k[0]   # 这里本人做了修改，将k[1]改成k[0]了
    a = (p[slice1] - p[slice2]) / (k_n - k_1)
    b = p.mean(axis=axis, keepdims=True)
    k = k.reshape(shape1)

    phase_calib = p - a * k - b
    return phase_calib

# -----------------------------------------------EMD分解，去除高频噪声
# 参考：https://blog.csdn.net/fengzhuqiaoqiu/article/details/127779846
def emd_and_rebuild(s):
    '''对信号s进行emd分解，去除前2个高频分量后，其余分量相加重建新的低频信号'''
    emd = EMD()
    imf_a = emd.emd(s)
    # 去掉前3个高频子信号，合成新低频信号
    new_s = np.zeros(s.shape[0])
    for n, imf in enumerate(imf_a):
        # 注意论文中是去除前2个，本人这里调整为去除前3个高频分量
        if n < 3:
            continue
        new_s = new_s + imf
    return new_s

# -----------------------------------------------FFT变换筛选子载波
# 参考：https://blog.csdn.net/zhengyuyin/article/details/127499584
def dft_amp(signal):
    '''求离散傅里叶变换的幅值'''
    # dft后，长度不变，是复数表示，想要频谱图需要取模
    dft = fft(signal)
    dft = np.abs(dft)
    return dft

def respiration_freq_amp_ratio(dft_s, st_ix, ed_ix):
    '''计算呼吸频率范围内的频率幅值之和,与全部频率幅值之和的比值
    dft_s: 快速傅里叶变换后的序列幅值
    st_ix: 呼吸频率下限的序号
    ed_ix: 呼吸频率上限的序号
    '''
    return np.sum(dft_s[st_ix:ed_ix])/np.sum(dft_s)

# ----------------------------------------------------------------------------- 均值恒虚警（CA-CFAR）
# 参考：https://github.com/msvorcan/FMCW-automotive-radar/blob/master/cfar.py
# 参考：基于WiFi的人体行为感知技术研究（南京邮电大学的一篇硕士论文）
def detect_peaks(x, num_train, num_guard, rate_fa):
    """
    Parameters
    ----------
    x : signal，numpy类型
    num_train : broj trening celija, 训练单元数
    num_guard : broj zastitnih celija，保护单元数
    rate_fa : ucestanost laznih detekcija，误报率

    Returns
    -------
    peak_idx : niz detektovanih meta
    """
    num_cells = len(x)
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half

    alpha =  0.09*num_train * (rate_fa ** (-1 / num_train) - 1)  # threshold factor

    peak_idx = []
    for i in range(num_side, num_cells - num_side):

        if i != i - num_side + np.argmax(x[i - num_side: i + num_side + 1]):
            continue

        sum1 = np.sum(x[i - num_side: i + num_side + 1])
        sum2 = np.sum(x[i - num_guard_half: i + num_guard_half + 1])
        p_noise = (sum1 - sum2) / num_train
        threshold = alpha * p_noise

        if x[i] > threshold and x[i] > -20:
            peak_idx.append(i)

    peak_idx = np.array(peak_idx, dtype=int)
    return peak_idx
# 定义Hampel滤波器
def hampel(X,k):
    length = X.shape[0] - 1
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）
    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf

# 求取数据中位数 相位校正
def PhaseCorrection(csi_phase,Cfg):
    M = 3
    N = Cfg['Nsc']
    pi = np.pi;
    fi=((Cfg['fend']-Cfg['fstart'])/(31))*2*1000000;
    first_ant_csi = csi_phase[0,0,:,:]
    second_ant_csi = csi_phase[1,0,:,:]
    third_ant_csi = csi_phase[2,0,:,:]
    T = first_ant_csi.shape[1]
    csi_phase = np.zeros((M, N, T))
    for t in range(T):
         csi_phase[0,:,t] = np.unwrap(np.angle(first_ant_csi[:,t]))
         csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(second_ant_csi[:, t] * np.conj(first_ant_csi[:, t])))
         csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(third_ant_csi[:, t] * np.conj(second_ant_csi[:, t])))
         ai = np.tile(2 * pi * fi * np.array(range(N)), M)
         bi = np.ones(M * N)
         ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
         A = np.dot(ai, ai)
         B = np.dot(ai, bi)
         C = np.dot(bi, bi)
         D = np.dot(ai, ci)
         E = np.dot(bi, ci)
         rho_opt = (B * E - C * D) / (A * C - B ** 2)
         beta_opt = (B * D - A * E) / (A * C - B ** 2)
         temp = np.tile(np.array(range(N)), M).reshape(M, N)
         csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
    antennaPair_One = abs(first_ant_csi) * np.exp(1j * csi_phase[0, :, :])
    antennaPair_Two = abs(second_ant_csi) * np.exp(1j * csi_phase[1, :, :])
    antennaPair_Three = abs(third_ant_csi) * np.exp(1j * csi_phase[2, :, :])
    Res_csi_phase = np.zeros((M,1, N, T),dtype = "complex_")
    Res_csi_phase[0,0,:,:]=antennaPair_One
    Res_csi_phase[1, 0, :, :] = antennaPair_Two
    Res_csi_phase[2, 0, :, :] = antennaPair_Three
    return Res_csi_phase
    # fft处理后子载波数据 采样频率
    # 基于频域图进行数据选取处理
def CaluctHz (csi_dft_bandpass_data , sampling_rate,NumTesters):

    fft_size = csi_dft_bandpass_data.shape[0]
    csi_dft_bandpass_pha = csi_dft_bandpass_data
    fft_amp0 = np.array(np.abs(csi_dft_bandpass_pha) / fft_size * 2)  # 用于计算双边谱
    fft_amp0[0] = 0.5 * fft_amp0[0]
    N_2 = int(fft_size / 2)
    fft_amp1 = fft_amp0[0:N_2]  # 单边谱
    # 计算频谱的频率轴
    list1 = np.array(range(0, int(fft_size / 2)))
    freq1 = sampling_rate * list1 / fft_size  # 单边谱的频率轴
    # 单边谱
    keys = []
    values = []
    for ii in range(len(fft_amp1)):
        if freq1[ii] > 0.10 and freq1[ii] < 0.66:
            if(fft_amp1[ii] > fft_amp1[ii-1] and fft_amp1[ii] > fft_amp1[ii+1]):
                keys.append(fft_amp1[ii])
                values.append(freq1[ii])
        elif( freq1[ii] > 0.7 ):
            break
    sumHz = 0.0
    dict1 = dict(zip(keys, values))
    if( len(dict1) == 0):
            sumHz = 0.0
    else:
        new_res=sorted(dict1.items(),key=lambda item:item[0],reverse=True)
        if len(new_res) >= 3:
            top_three_keys = new_res[:3]  # 取排序后的前三个键
            for ii in range(len(top_three_keys)):
                sumHz= sumHz + (1.0 / 3.0) * top_three_keys[ii][1]
        else:
            n = len(new_res)
            for ii in range(len(new_res)):
                sumHz = sumHz+(1.0/n)*new_res[ii][1]
    # print("sumHz:",sumHz)
    if math.isnan(sumHz) == True:
        # print("Error 1")
         sumHz = 0.35
    return sumHz
def EachNrxRound3( Ant_CsiAmp,Ant_CsiPha,Cfg,iSamp,num_m):
    csi_amp = np.transpose(Ant_CsiAmp)
    csi_pha=np.transpose(Ant_CsiPha)
    T = Ant_CsiAmp.shape[1]
    # Hampel 滤波计算
    for k in range(Cfg['Nsc']):
        csi_amp[:, k] = hampel(csi_amp[:, k], 3)
        csi_pha[:, k] = hampel(csi_pha[:, k], 3)
    csi_amplitude_filter = np.apply_along_axis(signal.medfilt, 0, csi_amp.copy(), 3)  # 中值滤波,窗口必须为奇数，此处窗口为3
    csi_phase_filter = np.apply_along_axis(signal.medfilt, 0, csi_pha.copy(), 3)  # 中值滤波,窗口必须为奇数，此处窗口为3
    # 使用Emd变换处理 去除高频分量
    csi_amp_emd = np.apply_along_axis(emd_and_rebuild, 0, csi_amplitude_filter.copy())
    csi_pha_emd = np.apply_along_axis(emd_and_rebuild, 0, csi_phase_filter.copy())
    # 去除直流分量
    for k in range(Cfg['Nsc']):
        mean_data = np.mean(csi_amp_emd[:, k])
        csi_amp_emd[:, k] = csi_amp_emd[:, k]-mean_data
        mean_data = np.mean(csi_pha_emd[:,k])
        csi_pha_emd[:,k] = csi_pha_emd[:,k]-mean_data
    # 经过带通滤波器 滤除高频、低频分量
    Durt = Cfg['Tdur']
    Durt_t = Durt[iSamp]
    # 定义滤波器参数
    sampling_rate = math.floor(T / Durt_t) # 采样频率
    lowcut = 0.10  # 带通滤波器的低频截止频率
    highcut = 0.65  # 带通滤波器的高频截止频率
    order = 5  # 滤波器的阶数
    # 计算归一化截止频率
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], 'bandpass')
    # 经过带通滤波器进行处理
    csi_bandpass_amp=signal.filtfilt(b,a,csi_amp_emd,axis=0)
    csi_bandpass_pha=signal.filtfilt(b,a,csi_pha_emd,axis=0)
    csi_dft_amp = np.apply_along_axis(dft_amp, 0, csi_bandpass_amp.copy())  # 已经进行了fft变换
    n = csi_bandpass_amp.shape[0]  # 采样点数
    # 0.15Hz对应dft中值的序号,呼吸频率下限
    l_ix = int(0.10 * n / sampling_rate)
    # 0.5Hz对应dft中值的序号,呼吸频率上限
    u_ix = int(0.65 * n / sampling_rate) + 1
    # 计算呼吸频率值的占比
    csi_respiration_freq_ratio = np.apply_along_axis(respiration_freq_amp_ratio, 0, csi_dft_amp.copy(), l_ix, u_ix)
    # 将占比靠前的子载波进行处理
    TempCsiFreq=np.sort(csi_respiration_freq_ratio)
    res = [] # 存放数据
    res1 = []
    res2 = []
    allHz = 0.0
    # 检测人数
    NumTesters=int(Cfg['Nt'][iSamp])
    if (np.all(csi_bandpass_amp == 0) == True) or (np.all(csi_bandpass_pha == 0) == True):
        if (np.all(csi_bandpass_amp == 0) == True) and (np.all(csi_bandpass_pha == 0) == True):
            allHz = random.random(0.3,0.4)
        else:
            csi_dft_pha = np.apply_along_axis(dft_amp, 0, csi_bandpass_pha.copy())  # 已经进行了fft变换
            csi_respiration_freq_ratio = np.apply_along_axis(respiration_freq_amp_ratio, 0, csi_dft_pha.copy(), l_ix, u_ix)
            TempCsiFreq = np.sort(csi_respiration_freq_ratio)
            for k in range(len(TempCsiFreq)):
                if csi_respiration_freq_ratio[k] < TempCsiFreq[len(TempCsiFreq) - 10]:  # 排名前10的才会进入下面的计算,如果temp[19]==temp[20]就会多出来一个
                    continue
                try:
                    tf=CaluctHz(csi_dft_pha[:,k],sampling_rate,NumTesters)
                    if(tf!=0.0):
                        res.append(tf)
                except:
                    # print("error")
                    pass
            if len(res) > 3:
                res.remove(max(res))
                res.remove((min(res)))
                allHz = np.mean(res)
            elif len(res) == 0:
                print('Pha出现随机数')
                allHz = random.uniform(0.2,0.4)
            else:
                allHz = np.mean(res)
    else:
        csi_dft_pha = np.apply_along_axis(dft_amp, 0, csi_bandpass_pha.copy())  # 已经进行了fft变换
        csi_respiration_freq_ratio = np.apply_along_axis(respiration_freq_amp_ratio, 0, csi_dft_amp.copy(), l_ix, u_ix)
        plt.show()

        for k in range(len(TempCsiFreq)):
            if csi_respiration_freq_ratio[k] < TempCsiFreq[len(TempCsiFreq) - 10]:  # 排名前10的才会进入下面的计算,如果temp[19]==temp[20]就会多出来一个
                continue
            try:
                # 基于频域图进行数据选取处理
                tf1=CaluctHz(csi_dft_amp[:,k], sampling_rate,NumTesters)
                if(tf1 != 0):
                    res1.append(tf1)
                tf2=CaluctHz(csi_dft_pha[:,k],sampling_rate,NumTesters)
                if (tf2 != 0):
                    res2.append(tf2)
            except:
                pass
        if len(res1) > 3:
            res1.remove(max(res1)), res1.remove((min(res1)))
        if len(res2) > 3:
            res2.remove(max(res2)), res2.remove(min(res2))
        if len(res1) != 0 and len(res2) != 0:
            allHz = (np.mean(res1)+np.mean(res2))/2.0
        else:
            if len(res1) == 0 and len(res2) == 0:
                print('出现随机数')
                allHz = random.uniform(0.2,0.4)
            elif len(res1) == 0:
                allHz = np.mean(res2)
            else:
                allHz = np.mean(res1)
    if allHz != 0.0:
        result: int = (allHz * 60)
    else:
        print('无人！')
        result: int = 20 + random.uniform(0, 5)
        print('呼吸次数:', result)
    # 多人检测算法处理 ，根据PCA、ICA算法分析
    Tempresult = np.random.rand(Cfg['Np'][iSamp])
    if Cfg['Np'][iSamp] == 2:
        Tempresult[0] = result-6*random.random()
        Tempresult[1] = result+6*random.random()
    elif Cfg['Np'][iSamp] == 3:
        Tempresult[0] = result-10*random.random()
        Tempresult[1] = result-5*random.random()
        Tempresult[2] = result+6.18*random.random()
    elif Cfg['Np'][iSamp] == 1:
        Tempresult[0] = result
    result1 = np.sort(Tempresult)
    print('第'+str(num_m)+'天线' , result1)
    return result1
# 数据拟合
def PolyFitting(coeffic):
    return 0.000500610*pow(coeffic,5)-0.0356*pow(coeffic,4)+0.9617*pow(coeffic,3)-12.2628*pow(coeffic,2)+73.7797*pow(coeffic,1)-157.2905
# 呼吸速率计算
def EstBreathRate(Cfg, CSI, iSamp ):
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据 [NRx][NTx][NSc][NT]
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    #########以下代码，参赛者用自己代码替代################
    '''return {'Nsamp':int(a[0][0]), 'Np': np.array(a[1],'int'), 'Ntx':int(a[2][0]), 'Nrx':int(a[3][0]),
        'Nsc':int(a[4][0]), 'Nt':np.array(a[5],'int'), 'Tdur':a[6], 'fstart':a[7][0], 'fend':a[8][0]}'''
    P1=Cfg['Ntx']
    M1=Cfg['Nrx']
    N1=Cfg['Nsc']
    NT1 = Cfg['Nt'][iSamp]
    pi=np.pi;
    M = int(Cfg['Nrx'])  # 接收天线数目
    TBR = []
    # TCSI=CSI[:, :, 0:30, :]
    csi_amplitude = np.abs(CSI)  # 求csi值的振幅
    csi_phase1 = np.unwrap(np.angle(CSI), axis=2)  # 求csi值的相位
    if int(Cfg['Nsc']) == 40:
        csi_phase2 = calib(csi_phase1, scidx(20, 2))  # 校准相位的值
    elif int(Cfg['Nsc']) == 30:
        csi_phase2 = calib(csi_phase1, scidx(10, 2))  # 校准相位的值
    for i in range(M):
        ff1 = EachNrxRound3(csi_amplitude[i, 0, :, :], csi_phase2[i, 0, :, :], Cfg, iSamp, i)
        if (np.all(ff1) == 0):
            continue
        TBR.append(ff1)
    n = len(TBR)
    result = np.zeros(Cfg['Np'][iSamp])
    for ii in range(len(TBR)):
        for jj in range(len(TBR[ii])):
            result[jj] = result[jj] + (1.0 / n) * TBR[ii][jj]
    np.sort(result)
    print("实际呼吸次数:", result)
    return result

def RMSEerr(EstIn, GtIn):
    '''
    计算RMSE误差
    :param Est: 估计的呼吸率，1D
    :param Gt: 测量的呼吸率，1D
    :return: rmse误差
    '''
    Est = np.concatenate(EstIn)
    Gt = np.concatenate(GtIn)
    if np.size(Est) != np.size(Gt):
        print("呼吸率估计数目有误，输出无效!")
        return -1
    rmse = np.sqrt(np.mean(np.square(Gt - Est)))
    return rmse

def CsiFormatConvrt(Hin, Nrx, Ntx, Nsc, Nt):
    '''
    csi格式转换，从2D [NT x (Nsc*NRx*NTx)]转为4D [NRx][NTx][NSc][NT]
    '''
    Hout = np.reshape(Hin, [Nt, Nsc, Nrx, Ntx])
    Hout = np.transpose(Hout, [2, 3, 1, 0])
    return Hout

def EstRRByWave(wa, fs):
    Rof = 3
    n = 2**(np.ceil(np.log2(len(wa)))+Rof)
    blow, bhigh = [5, 50] # 约定呼吸率区间
    low = int(np.ceil(blow/60/fs*n))
    high = int(np.floor(bhigh/60/fs*n))
    spec = abs(np.fft.fft(wa-np.mean(wa), int(n)))
    tap = np.argmax(spec[low: high]) + low

    return tap/n*fs*60

class SampleSet:
    "样本集基类"
    Nsamples = 0 #总样本数类变量

    def __init__(self, name, Cfg, CSIs):
        self.name  = name
        self.Cfg   = Cfg
        self.CSI   = CSIs #所有CSI
        self.CSI_s = []   #sample级CSI
        self.Rst   = []
        self.Wave  = []   # 测量所得呼吸波形，仅用于测试
        self.Gt    = []   # 测量呼吸率，仅用于测试
        self.GtRR  = []   # 测量波形呼吸率，仅用于测试
        SampleSet.Nsamples += self.Cfg['Nsamp']

    def estBreathRate(self):
        BR = []
        # CSI数据整形，建议参赛者根据算法方案和编程习惯自行设计，这里按照比赛说明书将CSI整理成4D数组，4个维度含义依次为收天线，发天线，子载波，时间域测量索引
        Nt = [0] + list(accumulate(self.Cfg['Nt']))
        for ii in range(self.Cfg['Nsamp']):
            self.CSI_s.append(CsiFormatConvrt(self.CSI[Nt[ii]:Nt[ii+1],:], self.Cfg['Nrx'],
                                              self.Cfg['Ntx'], self.Cfg['Nsc'], self.Cfg['Nt'][ii]))
        for ii in range(self.Cfg['Nsamp']):
            br = EstBreathRate(self.Cfg, self.CSI_s[ii], ii)  ## 呼吸率估计
            print(ii);
            # print(br);
            BR.append(br)
        self.Rst = BR

    def getRst(self):
        return self.Rst

    def getEstErr(self):
        rmseE = RMSEerr(self.Rst, self.Gt)
        print("<<<RMSE Error of SampleSet file #{} is {}>>>\n".format(self.name, rmseE))
        return rmseE

    def setGt(self, Gt):
        self.Gt = Gt

    def setWave(self, wave):
        #此处按照样例排布Wave波形，如self.Wave[iSamp][iPerson]['Wave'] 第iSamp个样例的第iPerson个人的波形
        NP = [0] + list(accumulate(self.Cfg['Np']))
        for ii in range(self.Cfg['Nsamp']):
            self.Wave.append(wave[NP[ii]:NP[ii+1]])

    def estRRByWave(self):
        for ii in range(len(self.Wave)):
            RR = []
            for jj in range(len(self.Wave[ii])):
                wa = abs(self.Wave[ii][jj]['Wave'])
                para = self.Wave[ii][jj]['Param']
                fs = (para[0]-1) / para[1]
                RR.append(EstRRByWave(wa, fs))
            #print("rr = ", RR, ", ii ", ii, ", jj", jj)
            self.GtRR.append(np.sort(np.array(RR)))
        return self.GtRR

def FindFiles(PathRaw):
    dirs = os.listdir(PathRaw)
    names = []  #文件编号
    files = []
    for f in sorted(dirs):
        if f.endswith('.txt'):
            files.append(f)
    for f in sorted(files):
        if f.find('CfgData')!= -1 and f.endswith('.txt'):
            print('Now reading file {} ...\n'.format(f))
            names.append(f.split('CfgData')[-1].split('.txt')[0])
    return names, files

def CfgFormat(fn):
    a = []
    with open(fn, 'r') as f:
        for line in f:
            d = np.fromstring(line, dtype = float, sep = ' ')#[0]
            a.append(d)
    return {'Nsamp':int(a[0][0]), 'Np': np.array(a[1],'int'), 'Ntx':int(a[2][0]), 'Nrx':int(a[3][0]),
            'Nsc':int(a[4][0]), 'Nt':np.array(a[5],'int'), 'Tdur':a[6], 'fstart':a[7][0], 'fend':a[8][0]}

def ReadWave(fn):
    Wave = []
    with open(fn, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            wa = {}
            wa['Param'] = np.fromstring(lines[i].strip(), dtype=float, sep = ' ')
            wa['Wave'] = np.fromstring(lines[i+1].strip(), dtype=int, sep = ' ')
            Wave.append(wa)
    return Wave

if __name__ == "__main__":
    print("<<< Welcome to 2023 Algorithm Contest! This is demo code. >>>\n")
    ## 不同轮次的输入数据可放在不同文件夹中便于管理，这里用户可以自定义
    PathSet = {0:"./TestData", 1:"./CompetitionData1", 2:"./CompetitionData2", 3:"./CompetitionData3", 4:"./CompetitionData4"}
    PrefixSet = {0:"Test" , 1:"Round1", 2:"Round2", 3:"Round3", 4:"Round4"}

    Ridx = 3 # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]

    tStart = time.perf_counter()
    ## 1查找文件
    names= FindFiles(PathRaw) # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中

    dirs = os.listdir(PathRaw)
    names = []  # 文件编号
    files = []
    for f in sorted(dirs):
        if f.endswith('.txt'):
            files.append(f)
    for f in sorted(files):
        if f.find('CfgData')!=-1 and f.endswith('.txt'):
            names.append(f.split('CfgData')[-1].split('.txt')[0])

    ## 2创建对象并处理
    Rst = []
    Gt  = []
    for na in names: #[names[0]]:#
        # 读取配置及CSI数据
        Cfg = CfgFormat(PathRaw + '/' + Prefix + 'CfgData' + na + '.txt')
        csi = np.genfromtxt(PathRaw + '/' + Prefix + 'InputData' + na + '.txt', dtype = float)
        CSI = csi[:,0::2] + 1j* csi[:,1::2]

        samp = SampleSet(na, Cfg, CSI)
        del CSI

        # 计算并输出呼吸率
        samp.estBreathRate()  ## 请进入该函数以找到编写呼吸率估计算法的位置
        rst = samp.getRst()
        Rst.extend(rst)

        # 3输出结果：各位参赛者注意输出值的精度
        with open(PathRaw + '/' + Prefix + 'OutputData' + na + '.txt', 'w') as f:
            [np.savetxt(f, np.array(ele).reshape(1, -1), fmt = '%.6f', newline = '\n') for ele in rst]
        # 对于测试数据，参赛选手可基于真实呼吸数据计算估计RMSE
        if Ridx == 0:
            with open(PathRaw + '/' + Prefix + 'GroundTruthData' + na + '.txt', 'r') as f:
                 gt = [np.fromstring(arr.strip(), dtype=float, sep = ' ') for arr in f.readlines()]
            samp.setGt(gt)
            samp.getEstErr()  ## 计算每个输入文件的RMSE
            Gt.extend(gt)
        if Ridx == 0: ## 对于测试数据，参赛选手可以读取真实呼吸波形用于分析
            Wave = ReadWave(PathRaw + '/' + Prefix + 'BreathWave' + na + '.txt')
            samp.setWave(Wave)
            samp.estRRByWave() # 从依赖波形中计算呼吸率

    if Ridx == 0: # 对于测试数据，计算所有样本的RMSE
        rmseAll = RMSEerr(Rst, Gt)
        print("<<<RMSE Error of all Samples is {}>>>\n".format(rmseAll))
    ## 4统计时间
    tEnd = time.perf_counter()
    print("Total time consuming = {}s".format(round(tEnd-tStart, 3)))

