# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import copy
from scipy.io import savemat, loadmat
import multiprocessing
import tqdm
import json

TEST_DIR = Path(__file__).parent.parent / 'tests'
DIR = Path(__file__).parent.parent
sys.path.append(str(DIR))

from gnpy.core.info import create_input_spectral_information, create_input_spectral_information_various
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
# logger.debug('All packages are imported. Logger is initialized.')
# clear TEST_DIR / 'data' / 'temp'
GlobalControl.clear_folder(TEST_DIR / 'data' / 'temp') # TODO: clear folder
import ast

def str_to_list(string):
    """将字符串转换为列表"""
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError) as e:
        print(f"转换错误: {e}")
        return None


def dict_to_json(data):
    """将字典转换为 JSON 字符串"""
    try:
        json_string = json.dumps(data, ensure_ascii=False)
        return json_string
    except (TypeError, ValueError) as e:
        print(f"转换错误: {e}")
        return None

def json_to_dict(json_string):
    """将 JSON 字符串转换为字典"""
    try:
        data = json.loads(json_string)
        return data
    except (json.JSONDecodeError, TypeError) as e:
        print(f"解析错误: {e}")
        return None


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
                flag_attn_profile=True, cr_peak=1.0, ns=None):
    if flag_attn_profile:
        raman_info_path = TEST_DIR / 'data' / 'temp' / f'raman_fiber_config_attenprofile_{time.time()}_{ind}.json'
    else:
        raman_info_path = TEST_DIR / 'data' / 'temp' / f'raman_fiber_config_{time.time()}_{ind}.json'
    raman_fiber_config(power_list, frequency_list, direction_list, raman_info_path,
                       fiber_length=fiber_length, attn_profile=flag_attn_profile,
                         cr_peak=cr_peak, ns=ns)

    # spectral information generation
    eqpt_params = load_json(TEST_DIR / 'data' / 'eqpt_config_copy.json')
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

    return p_signal, p_ase, p_nli


def raman_transmit(raman_pump_power, bool_plot=False):
    """Transimission with Raman amplifier

    Args:
        raman_pump_power (list): in Watt, len(raman_pump_power) == 6.
        bool_plot (bool): determine whether to plot or not.

    Return:
        net_gain (ndarray): in dB.
    """
    freq_start = 186.0e12
    freq_end = 196.0e12
    baud_rate = 200e9
    spacing = 200e9
    si_config(f_min=freq_start, f_max=freq_end, baud_rate=baud_rate, spacing=spacing)
    channel_num = automatic_nch(freq_start, freq_end, spacing)
    
    logger.debug(f'channel number = {channel_num}')

    launch_power = 4  # dBm/ch
    launch_power_array = launch_power * np.ones(channel_num)
    launch_power_array = dbm2w(launch_power_array)
    # launch_power_array = dbm2w(np.random.randint(-5, 5, channel_num))
    logger.debug(f'launch power = {launch_power} dBm/ch')

    qot_tool = 'no_nli'  # 'GGN_partial'

    power_list = np.array(raman_pump_power)
    wavelength_list = np.array([1513., 1496., 1477., 1458., 1432., 1420.])
    freq_list = 299792458 / (wavelength_list * 1e-9)
    direction_list = [-1, -1, -1, -1, -1, -1]

    signal_lin, ase_lin, nli_lin = propagation(power_list, freq_list, direction_list,
                                               launch_power_array, qot_tool=qot_tool, flag_attn_profile=True,
                                               cr_peak=0.0003841, ns=0.0)

    signal_dbm = w2dbm(np.array(signal_lin))
    net_gain = signal_dbm - w2dbm(launch_power_array)
    logger.info('Net gain [dB]:' + ' '.join([f'{x:.1f}' for x in net_gain]))

    if bool_plot:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(freq_start, freq_end, channel_num)/1e12, signal_dbm, label='Receive power w/ RA')
        ax.plot(np.linspace(freq_start, freq_end, channel_num)/1e12, w2dbm(launch_power_array), label='Launch power')
        ax.set_xlabel('Frequency [THz]')
        ax.set_ylabel('Signal power [dBm]')
        ax.grid()
        plt.show()
    return net_gain


def raman_transmit_dict_input(input_str):

    """Transimission with Raman amplifier

    Args:
        1 input string, describing a dict with a list of raman_pump_power (list): in Watt, len(raman_pump_power) == 6.
        and a bool: determine whether to plot or not.

    Return:
        net_gain (ndarray): in dB.
    """
    input = json_to_dict(input_str)
    raman_pump_power = str_to_list(input['raman_pump_power'])
    bool_plot = input['bool_plot']
    freq_start = 186.0e12
    freq_end = 196.0e12
    baud_rate = 200e9
    spacing = 200e9
    si_config(f_min=freq_start, f_max=freq_end, baud_rate=baud_rate, spacing=spacing)
    channel_num = automatic_nch(freq_start, freq_end, spacing)
    
    logger.debug(f'channel number = {channel_num}')

    launch_power = 4  # dBm/ch
    launch_power_array = launch_power * np.ones(channel_num)
    launch_power_array = dbm2w(launch_power_array)
    # launch_power_array = dbm2w(np.random.randint(-5, 5, channel_num))
    logger.debug(f'launch power = {launch_power} dBm/ch')

    qot_tool = 'no_nli'  # 'GGN_partial'

    power_list = np.array(raman_pump_power)
    wavelength_list = np.array([1513., 1496., 1477., 1458., 1432., 1420.])
    freq_list = 299792458 / (wavelength_list * 1e-9)
    direction_list = [-1, -1, -1, -1, -1, -1]

    signal_lin, ase_lin, nli_lin = propagation(power_list, freq_list, direction_list,
                                               launch_power_array, qot_tool=qot_tool, flag_attn_profile=True,
                                               cr_peak=0.0003841, ns=0.0)

    signal_dbm = w2dbm(np.array(signal_lin))
    net_gain = signal_dbm - w2dbm(launch_power_array)
    logger.info('Net gain [dB]:' + ' '.join([f'{x:.1f}' for x in net_gain]))

    if bool_plot:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(freq_start, freq_end, channel_num)/1e12, signal_dbm, label='Receive power w/ RA')
        ax.plot(np.linspace(freq_start, freq_end, channel_num)/1e12, w2dbm(launch_power_array), label='Launch power')
        ax.set_xlabel('Frequency [THz]')
        ax.set_ylabel('Signal power [dBm]')
        ax.grid()
        plt.show()
    return net_gain


if __name__ == '__main__':
    # 建模：完成从 raman_pump_power 到 net_gain的映射
    # raman_pump_power = [0.04, 0.03, 0.02, 0.12, 0.3, 0.3]  # 不要设置的太高(推荐<0.25)，len==6
    # net_gain = raman_transmit(raman_pump_power, bool_plot=True)
    # logger.info('完成！')

    input_data = {"raman_pump_power":"[0.01, 0.04, 0.1, 0.05, 0.2, 0.15]", "bool_plot":True}
    input_data = dict_to_json(input_data)
    ## 验证
    raman_transmit_dict_input(input_data)


