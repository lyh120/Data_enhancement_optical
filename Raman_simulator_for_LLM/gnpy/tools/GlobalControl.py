import logging
import time
import coloredlogs
import os

class GlobalControl(object):
    # logger = None
    def __init__(cls, logger=None):
        cls.logger = logger

    @classmethod
    def init_logger(cls, name='logger' + time.strftime('%Y-%m-%d',time.localtime(time.time())),
                    level=1, color_set_mode='modified', file_output_dir=None):
        cls.logger = MyLogging(name, level, color_set_mode, file_output_dir=file_output_dir).logger
        return cls.logger

    @classmethod
    def clear_folder(cls, folder_path):
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.json'):
                os.remove(file_path)
        cls.logger.debug(f'Path {folder_path} cleared!')


class MyLogging(object):
    logger = logging.Logger('default')
    def __init__(self, name, level=1, color_set_mode='modified', file_output_dir=None):
        self.logger = logging.Logger(name)
        fmt = '%(asctime)s [%(levelname)s] [%(name)s] %(filename)s[line:%(lineno)d] %(message)s'
        formater = logging.Formatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formater)
        self.logger.addHandler(ch)
        if file_output_dir is not None:
            file_handler = logging.FileHandler(file_output_dir)
            file_handler.setFormatter(formater)
            self.logger.addHandler(file_handler)
        if color_set_mode=='modified':
            coloredlogs.DEFAULT_FIELD_STYLES = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'},
                                                'levelname': {'color': 'green', 'bold': True}, 'request_id':{'color': 'yellow'},
                                                'name': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'threadName': {'color': 'yellow'}}
        coloredlogs.install(fmt=fmt, level=level, logger=self.logger)


if __name__ == '__main__':
    GlobalControl.init_logger(file_output_dir = r'scripts\20231014_exp_ofc_main_results\logtest.log')
    gc = GlobalControl.logger
    gc.debug('test debug')
    gc.info('test info')
    gc.warning('test warning')
