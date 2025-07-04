# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import time
from datetime import datetime
import random


TEST_DIR = Path(__file__).parent.parent / 'tests'
DIR = Path(__file__).parent.parent
sys.path.append(str(DIR))

from gnpy.core.info import create_input_spectral_information_various
from gnpy.core.elements import RamanFiber
from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import Simulation
from gnpy.tools.json_io import load_json, save_json
from gnpy.core.utils import automatic_nch, lin2db, db2lin, w2dbm, dbm2w
from gnpy.tools.GlobalControl import GlobalControl

try:
    logger = GlobalControl.logger
except:
    GlobalControl.init_logger('log'+datetime.now().strftime("%Y%m%d-%H%M%S"), 1, 'modified')
    logger = GlobalControl.logger
    logger.debug('All packages are imported. Logger is initialized.')
# clear TEST_DIR / 'data' / 'temp'
GlobalControl.clear_folder(TEST_DIR / 'data' / 'temp') # TODO: clear folder

import numpy as np
from gnpy.core.utils import lin2db, db2lin

class EDFA_SIMU():
    def __init__(self, gain, nf, freq_min=191.2875e12, freq_max=196.0875e12) -> None:
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.nf = nf
        self._tilt = 0  # in dB/THz
        self._gain = None
        self.set_gain(gain)

    def __add__(self, other):
        if not isinstance(other, EDFA_SIMU):
            raise TypeError("Unsupported operand type(s)")
        else:
            assert not(np.isscalar(self.gain) or np.isscalar(other.gain)), "Operands should be vectors."
            assert not(np.isscalar(self.nf) or np.isscalar(other.nf)), "Operands should be vectors."
            gain_temp = np.hstack((np.array(self.gain), np.array(other.gain)))
            nf_temp = np.hstack((np.array(self.nf), np.array(other.nf)))
            freq_min = min(self.freq_min, other.freq_min)
            freq_max = max(self.freq_max, other.freq_max)
            return EDFA_SIMU(gain_temp, nf_temp, freq_min, freq_max)

    def __call__(self, signal, ase, nli, baudrate):
        signal_o, ase_o, nli_o = self.prop(signal, ase, nli, baudrate)
        return signal_o, ase_o, nli_o

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self.set_gain(value)

    @property
    def gain_db(self):
        return lin2db(self._gain)
    
    @property
    def nf_db(self):
        return lin2db(self.nf)

    @property
    def tilt(self):
        return self._tilt
    
    @tilt.setter
    def tilt(self, value):
        self.set_tilt(value)

    def set_tilt(self, tilt_db_per_thz):
        self._tilt = tilt_db_per_thz
        self._update_gain()

    def set_gain(self, gain):
        if np.isscalar(gain):
            self._gain = gain
            self.channel_count = 1
        else:
            self._gain = np.array(gain)
            self.channel_count = len(gain)
        self._update_gain()

    def _update_gain(self):
        if self.channel_count == 1:
            return

        self.freq_array = np.linspace(self.freq_min, self.freq_max, self.channel_count)
        center_freq = (self.freq_max + self.freq_min) / 2
        freq_diff_thz = (self.freq_array - center_freq) / 1e12  # Convert to THz
        gain_tilt = self._tilt * freq_diff_thz
        
        mean_gain_db = np.mean(self.gain_db)
        new_gain_db = mean_gain_db + gain_tilt - np.mean(gain_tilt)
        self._gain = db2lin(new_gain_db)

    def calc_ase(self, baudrate):
        h = 6.62606930800080626e-34
        freq_ref = 192.0e12
        return self.nf * (self._gain - 1) * h * freq_ref * baudrate
    
    def prop(self, signal, ase, nli, baudrate):
        if np.isscalar(self._gain):
            self._gain = np.ones_like(signal) * self._gain
        if np.isscalar(self.nf):
            self.nf = np.ones_like(signal) * self.nf
        assert len(self._gain) == len(signal), "Length of EDFA gain should equal to signal."
        assert len(self.nf) == len(signal), "Length of EDFA NF should equal to signal."
        self._gain = np.array(self._gain)
        self.nf = np.array(self.nf)

        signal = signal * self._gain
        ase = ase * self._gain + self.calc_ase(baudrate)
        nli = nli * self._gain
        return signal, ase, nli


class EDFA_TENCENT_SIMU(EDFA_SIMU):
    def __init__(self, gain, tilt, saturation_power_db=23.5):
        self.channel_count = 64
        self.gain_list_db = [7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25]
        self.nf_list_db = [17.2, 15.2, 13.3, 11.6, 10.2, 9.1, 8.1, 7.3, 6.5, 6.1, 5.8, 5.5, 5.3, 6.2, 5.7, 5.5, 5.3]
        # 使用三次样条插值
        self.nf_interp = interpolate.CubicSpline(self.gain_list_db, self.nf_list_db, extrapolate=True)
        self.saturation_power_db = saturation_power_db

        super().__init__(gain, nf=None, freq_min=191.2875e12, freq_max=196.0875e12)
        
        self.set_tilt(tilt)

    def set_gain(self, gain):
        if not np.isscalar(gain):
            raise ValueError("Gain must be a scalar value")

        self._gain = np.ones(self.channel_count) * db2lin(gain)
        self._update_nf()
        self._update_gain()

    def set_tilt(self, tilt):
        if not np.isscalar(tilt):
            raise ValueError("Tilt must be a scalar value")
        self._tilt = tilt
        self._update_gain()
        self._update_nf()  # 重新计算nf

    def _update_nf(self):
        avg_gain_db = np.mean(lin2db(self._gain))
        self.nf = db2lin(self.nf_interp(avg_gain_db))

    def _update_gain(self):
        freq_array = np.linspace(self.freq_min, self.freq_max, self.channel_count)
        center_freq = (self.freq_max + self.freq_min) / 2
        freq_diff = freq_array - center_freq
        
        # Calculate gain tilt
        gain_tilt_db = self._tilt * freq_diff / (self.freq_max - self.freq_min)
        
        # Adjust gain to maintain the average
        avg_gain_db = np.mean(lin2db(self._gain))
        new_gain_db = avg_gain_db + gain_tilt_db - np.mean(gain_tilt_db)
        self._gain = db2lin(new_gain_db)

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self.set_gain(value)

    @property
    def tilt(self):
        return self._tilt
    
    @tilt.setter
    def tilt(self, value):
        self.set_tilt(value)

    def is_saturation(self, signal):
        return w2dbm(np.sum(signal)) > self.saturation_power_db

    def prop(self, signal, ase, nli, baudrate):
        if np.isscalar(self.nf):
            self.nf = np.ones_like(signal) * self.nf
        assert len(self._gain) == len(signal), "Length of EDFA gain should equal to signal."
        assert len(self.nf) == len(signal), "Length of EDFA NF should equal to signal."

        self._gain = np.array(self._gain)
        self.nf = np.array(self.nf)

        if self.is_saturation(signal * self._gain):
            saturated_gain_db = self.saturation_power_db - w2dbm(np.sum(signal))
            # logger.warning(f'输出功率饱和,设置增益{lin2db(np.mean(self._gain))}dB,实际增益{saturated_gain_db}')
            self.set_gain(saturated_gain_db)
            
        signal = signal * self._gain
        ase = ase * self._gain + self.calc_ase(baudrate)
        nli = nli * self._gain
        return signal, ase, nli



class EDFA_TENCENT_PA_SIMU(EDFA_TENCENT_SIMU):
    def __init__(self, gain, tilt, saturation_power_db=23.5):
        super().__init__(gain, tilt, saturation_power_db)

        self.gain_list_db = [15,33]
        self.nf_list_db = [6.5, 6.5]
        self.nf_interp = interpolate.CubicSpline(self.gain_list_db, self.nf_list_db, extrapolate=True)

        self._update_nf()


class MUX():
    def __init__(self, attn):
        # All params are in LINEAR units.
        self.attn = attn

    def __add__(self, other):
        if not isinstance(other, MUX):
            raise TypeError("Unsupported operand type(s)")
        else:
            assert not(np.isscalar(self.attn) or np.isscalar(other.attn)), "Operands should be vectors."
            attn_temp = np.hstack((np.array(self.attn), np.array(other.attn)))
            return MUX(attn_temp)

    @property
    def attn_db(self):
        return lin2db(self.attn)
    
    def __call__(self, signal, ase, nli):
        signal_o, ase_o, nli_o = self.prop(signal, ase, nli)
        return signal_o, ase_o, nli_o

    def prop(self, signal, ase, nli):
        if np.isscalar(self.attn):
            self.attn = np.ones_like(signal) * self.attn
        self.attn = np.array(self.attn)
        assert len(self.attn) == len(signal), "Length of MUX attn should equal to signal."
        signal = signal / self.attn
        ase = ase / self.attn
        nli = nli / self.attn
        return signal, ase, nli


class VOA():
    def __init__(self, attn):
        # All params are in LINEAR units.
        self.attn = attn

    @property
    def attn_db(self):
        return lin2db(self.attn)
    
    def __call__(self, signal, ase, nli):
        signal_o, ase_o, nli_o = self.prop(signal, ase, nli)
        return signal_o, ase_o, nli_o

    def prop(self, signal, ase, nli):
        if np.isscalar(self.attn):
            self.attn = np.ones_like(signal) * self.attn
        self.attn = np.array(self.attn)
        assert len(self.attn) == len(signal), "Length of MUX attn should equal to signal."
        signal = signal / self.attn
        ase = ase / self.attn
        nli = nli / self.attn
        return signal, ase, nli


def raman_fiber_config(power_list, frequency_list, direction_list, raman_info_path, fiber_length=80, attn_profile=False,
                       cr_peak=0.0003841, ns=None):
    if attn_profile:
        raman_info = load_json(TEST_DIR / 'data' / 'raman_fiber_config_attenprofile.json')
    else:
        raman_info = load_json(TEST_DIR / 'data' / 'raman_fiber_config.json')
    num_pump = len(power_list)
    raman_pumps_new = []

    for ind_pump in range(num_pump):
        pump_temp = {
            'power': power_list[ind_pump],
            'frequency': frequency_list[ind_pump],
            'propagation_direction': 'coprop' if direction_list[ind_pump] == 1 else 'counterprop'
        }
        raman_pumps_new.append(pump_temp)

    raman_info['operational']['raman_pumps'] = raman_pumps_new
    raman_info['params']['length'] = fiber_length

    cr = np.array(raman_info['params']['raman_efficiency']['cr'])
    cr_norm = cr / np.max(cr)
    cr = cr_norm * cr_peak
    raman_info['params']['raman_efficiency']['cr'] = cr.tolist()
    if ns is not None:
        raman_info['params']['raman_efficiency']['ns'] = ns

    save_json(raman_info, raman_info_path)


def si_config(f_min, f_max, baud_rate, spacing):
    eqpt_params = load_json(TEST_DIR / 'data' / 'eqpt_config_copy.json')
    spectral_info_params_new = eqpt_params['SI'][0]
    spectral_info_params_new['f_min'] = f_min
    spectral_info_params_new['f_max'] = f_max
    spectral_info_params_new['baud_rate'] = baud_rate
    spectral_info_params_new['spacing'] = spacing
    eqpt_params['SI'][0] = spectral_info_params_new
    save_json(eqpt_params, TEST_DIR / 'data' / 'eqpt_config_copy.json')


def propagation(power_list, frequency_list, direction_list, launch_power_arr, ase_arr=None, nli_arr=None, fiber_length=80, ind=0, qot_tool='GGN',
                flag_attn_profile=True, cr_peak=0.0003841, ns=None):
    if flag_attn_profile:
        raman_info_path = TEST_DIR / 'data' / 'temp' / f'raman_fiber_config_attenprofile_{time.time()}_{ind}.json'
    else:
        raman_info_path = TEST_DIR / 'data' / 'temp' / f'raman_fiber_config_{time.time()}_{ind}.json'
    raman_fiber_config(power_list, frequency_list, direction_list, raman_info_path,
                       fiber_length=fiber_length, attn_profile=flag_attn_profile,
                         cr_peak=cr_peak, ns=ns)

    # spectral information generation
    try_max_times = 10
    for i in range(try_max_times):
        try:
            eqpt_params = load_json(TEST_DIR / 'data' / 'eqpt_config_copy.json')  # TODO: 修改逻辑为取消文件读写
            break
        except:
            logger.warning(f'si_config 第{i+1}/{try_max_times}次尝试')
            time.sleep(random.uniform(0, 1))

    spectral_info_params = eqpt_params['SI'][0]
    spectral_info_params.pop('power_dbm')
    spectral_info_params.pop('power_range_db')
    spectral_info_params.pop('tx_osnr')
    spectral_info_params.pop('sys_margins')
    spectral_info_input = create_input_spectral_information_various(power_arr=launch_power_arr, nli_arr=nli_arr, ase_arr=ase_arr, **spectral_info_params)

    if qot_tool == 'GGN':
        sim_params = SimParams(**load_json(TEST_DIR / 'data' / 'sim_params_ggn.json'))
    elif qot_tool == 'GGN_partial':
        sim_params = SimParams(**load_json(TEST_DIR / 'data' / 'sim_params_ggn_partial.json'))
    elif qot_tool == 'GGN_partial_80ch':
        sim_params = SimParams(**load_json(TEST_DIR / 'data' / 'sim_params_ggn_partial_80ch.json'))
    elif qot_tool == 'GGN_partial_64ch_fieldtrial':
        sim_params = SimParams(**load_json(TEST_DIR / 'data' / 'sim_params_ggn_partial_64ch_fieldtrial.json'))
    elif qot_tool == 'GN_64ch_fieldtrial':
        sim_params = SimParams(**load_json(TEST_DIR / 'data' / 'sim_params_gn_64ch_fieldtrial.json'))
    elif qot_tool == 'GN':
        sim_params = SimParams(**load_json(TEST_DIR / 'data' / 'sim_params_gn.json'))
    elif qot_tool == 'no_nli':
        sim_params = SimParams(**load_json(TEST_DIR / 'data' / 'sim_params_no_nli.json'))
    else:
        logger.error('QoT tool 有误！')
        return
    Simulation.set_params(sim_params)
    fiber = RamanFiber(**load_json(raman_info_path))

    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = [carrier.power.signal for carrier in spectral_info_out.carriers]
    p_ase = [carrier.power.ase for carrier in spectral_info_out.carriers]
    p_nli = [carrier.power.nli for carrier in spectral_info_out.carriers]

    return np.array(p_signal), np.array(p_ase), np.array(p_nli)


def gn_occ(loading_status, gain_list, tilt_list, bool_verbose=True):
    """
    Propagation for C+L system with pure EDFA amplification
    NLI is estimated based on the INCOHERENT (G)GN model.
    """
    loading_status = loading_status[::-1]
    tilt_list = np.abs(tilt_list)
    
    ################
    # Signal define
    ################
    freq_start = 191.2875e12
    freq_end = 196.0875e12
    baud_rate = 63.9e9
    spacing = 75e9
    
    try_max_times = 10
    for i in range(try_max_times):
        try:
            si_config(f_min=freq_start, f_max=freq_end, baud_rate=baud_rate, spacing=spacing)  # TODO: 修改逻辑为取消文件读写
            break
        except:
            logger.warning(f'si_config 第{i+1}/{try_max_times}次尝试')
    
    channel_num = automatic_nch(freq_start, freq_end, spacing)

    # launch power config
    launch_power = -4 # dBm/ch
    center_channels = [26, 31, 36, 41, 46, 51]  # 最低频信道为编号1, 对应最高频为编号1情况下的[14,19,24,29,34,39]

    trx_osnr = 18  # dB
    # trx_osnr = np.full(channel_num, 60, dtype=float)
    # trx_osnr_set = [16, 16, 16, 16, 18, 18]
    # for center, osnr in zip(center_channels, trx_osnr_set):
    #     for i in range(-2, 3):  # 范围是 -2, -1, 0, 1, 2
    #         index = center + i
    #         if 1 <= index <= channel_num:  # 确保索引在有效范围内
    #             trx_osnr[index-1] = osnr

    launch_power_array = np.full(channel_num, -60) # 初始化所有信道功率为 -60 dBm

    for status, center in zip(loading_status, center_channels):
        if status == 1:
            for i in range(center-2, center+3):
                if 1 <= i <= channel_num:
                    launch_power_array[i-1] = launch_power

    # 将 dBm 转换为瓦特
    launch_power_array = dbm2w(launch_power_array)

    # 确定上波信道
    active_channels = []
    active_center_channels = []
    for i, status in enumerate(loading_status):
        if status == 1:
            center = center_channels[i]
            active_channels.extend(range(center-2, center+3))
            active_center_channels.extend([center])
    # logger.debug(f'Launch power configuration completed, total_ch_cum = {channel_num}, active_ch_num = {len(active_channels)}')


    ################
    # Sim. params. define
    ################
    qot_tool = 'GN_64ch_fieldtrial'  # WARN: 禁止修改.


    ################
    # Link define
    ################
    """
    Tx ─── mux ─── edfa_sh_ba ─── fiber[0] ─── edfa_ila_1 ─── fiber[1] ─── edfa_hz_pa ──┐
                                                                                        │
                                                                                        fiber[2]
                                                                                        │
    Rx ─── mux ─── edfa_sh_pa ─── fiber[4] ─── edfa_ila_2 ─── fiber[3] ─── edfa_hz_ba ──┘
    """

    ## Fiber define
    fiber = [117.7, 102.4, 1e-3, 102.4, 117.7]

    ## EDFA define
    edfa_sh_ba = EDFA_TENCENT_SIMU(gain=gain_list[0], tilt=tilt_list[0], saturation_power_db=24.5)
    edfa_ila_1 = EDFA_TENCENT_PA_SIMU(gain=gain_list[1], tilt=tilt_list[1])
    edfa_hz_pa = EDFA_TENCENT_PA_SIMU(gain=gain_list[2], tilt=tilt_list[2])
    edfa_hz_ba = EDFA_TENCENT_SIMU(gain=gain_list[3], tilt=tilt_list[3], saturation_power_db=24.5)
    edfa_ila_2 = EDFA_TENCENT_PA_SIMU(gain=gain_list[4], tilt=tilt_list[4])
    edfa_sh_pa = EDFA_TENCENT_PA_SIMU(gain=gain_list[5], tilt=tilt_list[5], saturation_power_db=8)

    ## MUX define
    mux = MUX(db2lin(7.5))

    ## VOA define
    voa_3db = VOA(db2lin(3))
    voa = VOA(db2lin(0.1))

    ## RA define
    # w/o RA
    power_list = []
    freq_list = []
    direction_list = []


    ################
    # Transmission
    ################
    # Tx
    sig_arr = launch_power_array
    ase_arr = sig_arr / db2lin(trx_osnr)
    nli_arr = np.zeros_like(sig_arr)

    # link
    sig_arr, ase_arr, nli_arr = mux(sig_arr, ase_arr, nli_arr)
    # sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = edfa_sh_ba(sig_arr, ase_arr, nli_arr, baud_rate)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = propagation(power_list, freq_list, direction_list,sig_arr, ase_arr=ase_arr, nli_arr=nli_arr,
                                            fiber_length=fiber[0],
                                            qot_tool=qot_tool, flag_attn_profile=False)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = edfa_ila_1(sig_arr, ase_arr, nli_arr, baud_rate)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = propagation(power_list, freq_list, direction_list,sig_arr, ase_arr=ase_arr, nli_arr=nli_arr,
                                            fiber_length=fiber[1],
                                            qot_tool=qot_tool, flag_attn_profile=False)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = edfa_hz_pa(sig_arr, ase_arr, nli_arr, baud_rate)

    sig_arr, ase_arr, nli_arr = propagation(power_list, freq_list, direction_list,sig_arr, ase_arr=ase_arr, nli_arr=nli_arr,
                                            fiber_length=fiber[2],
                                            qot_tool=qot_tool, flag_attn_profile=False)

    sig_arr, ase_arr, nli_arr = edfa_hz_ba(sig_arr, ase_arr, nli_arr, baud_rate)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = propagation(power_list, freq_list, direction_list,sig_arr, ase_arr=ase_arr, nli_arr=nli_arr,
                                            fiber_length=fiber[3],
                                            qot_tool=qot_tool, flag_attn_profile=False)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = edfa_ila_2(sig_arr, ase_arr, nli_arr, baud_rate)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = propagation(power_list, freq_list, direction_list,sig_arr, ase_arr=ase_arr, nli_arr=nli_arr,
                                            fiber_length=fiber[4],
                                            qot_tool=qot_tool, flag_attn_profile=False)
    sig_arr, ase_arr, nli_arr = voa_3db(sig_arr, ase_arr, nli_arr)
    sig_arr, ase_arr, nli_arr = edfa_sh_pa(sig_arr, ase_arr, nli_arr, baud_rate)
    sig_arr, ase_arr, nli_arr = mux(sig_arr, ase_arr, nli_arr)

    # Rx
    ase_arr += sig_arr / db2lin(trx_osnr)
    # 结果输出
    freq_list = np.linspace(freq_start, freq_end, channel_num)
    freq_points = []
    power_points = []
    osnr_points = []
    snrnli_points = []
    gsnr_points = []

    gsnr_db_list = []

    # 处理每个信道的数据
    for i in range(1, channel_num + 1):  # 从1开始，到channel_num
        freq = freq_list[i-1]  # 因为freq_list的索引从0开始
        sig = sig_arr[i-1]
        ase = ase_arr[i-1]
        nli = nli_arr[i-1]
        
        channel_start = freq - baud_rate/2
        channel_end = freq + baud_rate/2
        freq_points.extend([channel_start, channel_start, channel_end, channel_end])
        
        power_db = w2dbm(sig)
        power_points.extend([-60, power_db, power_db, -60])
        
        if i in active_center_channels:  # 只为上波的中心信道计算和存储OSNR, SNRNLI, GSNR
            osnr = lin2db(sig / ase)
            snrnli = lin2db(sig / nli)
            gsnr = lin2db(sig / (ase + nli))
            osnr_points.extend([0, osnr, osnr, 0])
            snrnli_points.extend([0, snrnli, snrnli, 0])
            gsnr_points.extend([0, gsnr, gsnr, 0])
            gsnr_db_list.append(gsnr)
        else:
            osnr_points.extend([np.nan, np.nan, np.nan, np.nan])
            snrnli_points.extend([np.nan, np.nan, np.nan, np.nan])
            gsnr_points.extend([np.nan, np.nan, np.nan, np.nan])
            if i in center_channels:
                gsnr_db_list.append(0.0)

    if bool_verbose:
        for center, status in zip(center_channels, loading_status):
            idx = center - 1  # 将信道编号转换为数组索引
            if status == 1:
                power = w2dbm(sig_arr[idx])
                osnr = lin2db(sig_arr[idx] / ase_arr[idx])
                snrnli = lin2db(sig_arr[idx] / nli_arr[idx])
                gsnr = lin2db(sig_arr[idx] / (ase_arr[idx] + nli_arr[idx]))
                logger.info(f"Channel {center}: Power = {power:.2f} dBm, OSNR = {osnr:.2f} dB, SNRNLI = {snrnli:.2f} dB, GSNR = {gsnr:.2f} dB")
            else:
                logger.info(f"Channel {center}: Inactive")
        logger.info(f"**** GSNR [dB] = {[f'{x:.2f}' for x in gsnr_db_list]} ****")

        # 绘制光谱图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(freq_points, power_points)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dBm)')
        ax.set_xlim(freq_start, freq_end)
        ax.set_ylim(max(power_points) - 3, max(power_points) + 1)  # 设置y轴范围，底部略低于-60dBm
        ax.grid(True)
        fig.tight_layout()

        # 绘制OSNR、SNRNLI和GSNR图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(freq_points, osnr_points, label='OSNR')
        ax.plot(freq_points, snrnli_points, label='SNRNLI')
        ax.plot(freq_points, gsnr_points, label='GSNR')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('dB')
        ax.set_title('OSNR, SNRNLI, and GSNR')
        ax.set_xlim(freq_start, freq_end)
        ax.set_ylim(np.nanmax(gsnr_points) - 5, max(np.nanmax(osnr_points), np.nanmax(snrnli_points)) + 2)
        
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    return gsnr_db_list[::-1]



if __name__ == '__main__':
    input = [1, 1, 1, 1, 1, 1, 22.8, 28.8, 18.2, 10.7, 25.8, 21.3, -0.8, -1.2, -1.3, -1.5, -0.2, -1.9]
    loading_status = input[0:6]
    gain_list = input[6:12]
    tilt_list = input[12:18]
    q_list = gn_occ(loading_status, gain_list, tilt_list, bool_verbose=False)
    print(q_list)