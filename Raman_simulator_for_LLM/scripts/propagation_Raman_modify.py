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

GlobalControl.init_logger('log'+datetime.now().strftime("%Y%m%d-%H%M%S"), 1, 'modified')
# logger = GlobalControl.logger
# logger.debug('All packages are imported. Logger is initialized.')
# clear TEST_DIR / 'data' / 'temp'
GlobalControl.clear_folder(TEST_DIR / 'data' / 'temp') # TODO: clear folder


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
        # logger.error('QoT tool 有误！')
        return
    Simulation.set_params(sim_params)
    fiber = RamanFiber(**load_json(raman_info_path))

    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = [carrier.power.signal for carrier in spectral_info_out.carriers]
    p_ase = [carrier.power.ase for carrier in spectral_info_out.carriers]
    p_nli = [carrier.power.nli for carrier in spectral_info_out.carriers]

    return p_signal, p_ase, p_nli


def raman_transmit(raman_pump_power):
    """Transimission with Raman amplifier

    Args: raman_pump_power (list of float): A list containing the Raman pump powers for each of the Raman pumps in the system, measured in Watts. The length of this list is expected to be 6, assuming a system with six Raman pumps, each potentially at a different wavelength to stimulate the Raman gain process.  
 
    Returns: net_gain_ave (numpy.ndarray): An array of floats representing the average net gain experienced by the optical signal across five discrete segments of the transmission link, measured in decibels (dB). The net gain takes into account both the Raman amplification and any residual losses in the fiber between amplifiers. The length of this array is 5, reflecting the evaluation of gain across five distinct regions of the transmission system.  
    
    Note: The exact calculation of the net gain in each segment depends on factors such as the Raman gain coefficient of the fiber, the pump wavelengths and powers, the fiber loss coefficient, and the distance between Raman amplifiers. This function abstracts these calculations, assuming a predefined model for the Raman amplification process and fiber losses. The returned net gain values are essential for assessing the overall performance of the transmission system, including its ability to maintain sufficient signal strength over long distances.
    """  
    # raman_pump_power = [float(item) for item in raman_pump_power_str.split(', ')]  
    
    freq_start = 186.0e12
    freq_end = 196.0e12
    baud_rate = 200e9
    spacing = 200e9
    si_config(f_min=freq_start, f_max=freq_end, baud_rate=baud_rate, spacing=spacing)
    channel_num = automatic_nch(freq_start, freq_end, spacing)
    # logger.debug(f'channel number = {channel_num}')

    launch_power = 4  # dBm/ch
    launch_power_array = launch_power * np.ones(channel_num)
    launch_power_array = dbm2w(launch_power_array)
    # launch_power_array = dbm2w(np.random.randint(-5, 5, channel_num))
    # logger.debug(f'launch power = {launch_power} dBm/ch')

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
    # logger.info('Net gain [dB]:' + ' '.join([f'{x:.1f}' for x in net_gain]))

    # if bool_plot:
    #     fig, ax = plt.subplots()
    #     ax.plot(np.linspace(freq_start, freq_end, channel_num)/1e12, signal_dbm, label='Receive power w/ RA')
    #     ax.plot(np.linspace(freq_start, freq_end, channel_num)/1e12, w2dbm(launch_power_array), label='Launch power')
    #     ax.set_xlabel('Frequency [THz]')
    #     ax.set_ylabel('Signal power [dBm]')
    #     ax.grid()
    #     plt.show()

    return net_gain

def raman_ave(raman_pump_power):

    """Transimission with Raman amplifier

    Args: Six float values representing the Raman pump powers.
    Returns: Five float values representing the average net gain across five segments.

    Note: Calculates the gain spectrum based on six Raman pump powers and returns the average net gain across five distinct transmission segments.
    """

    net_gain = raman_transmit(raman_pump_power)

    net_gain_ave = []
    for i in range(0, len(net_gain), 10):    
        group = net_gain[i:i+10]    
        net_gain_ave.append(sum(group) / len(group))

    return net_gain_ave


if __name__ == '__main__':
    # 建模：完成从 raman_pump_power 到 net_gain的映射
    raman_pump_power = [0.04, 0.03, 0.02, 0.12, 0.3, 0.3]  # 不要设置的太高(推荐<0.25)，len==6
    net_gain = raman_transmit(raman_pump_power)
    print(net_gain.shape)
