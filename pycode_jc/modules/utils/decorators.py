# coding:utf-8
import time
import logging

logger = logging.getLogger("COST_TIME")
fileHandler = logging.FileHandler("/srv/Data/projectLogs/cost_time.log")
fmt = '%(asctime)s - %(message)s'
formatter = logging.Formatter(fmt)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(logging.DEBUG)


def log_process_time(process_name):
    """
    记录函数处理时间
    :param process_name:
    :return:
    """

    def d(method_func):
        def inner(instance, *args, **kwargs):
            st = time.time()
            rtn = method_func(instance, *args, **kwargs)
            et = time.time()
            logger.info("cost:%s [%s]", format(et - st, '.4f'), process_name)
            #if et - st >= 0.001:
            #    logger.info("[%s] cost:%s", process_name, et - st)
            return rtn

        return inner

    return d
