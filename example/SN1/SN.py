import requests
import pickle
import configparser

def get_SN_peer():
    config = configparser.ConfigParser()
    config.read('./config.ini',encoding='utf-8')
    url = "http://%s:%s/"%(config.get('SEED', 'SEED_IP'),config.get('SEED', 'SEED_PORT'))
    #读取配置并转字典
    SN_config = dict(config.items('SG'))
    SN_config['port'] = int(SN_config['port'])
    data = pickle.dumps(SN_config)

    response = requests.request("POST", url, data=data)
    pic_set = pickle.loads(response.content)
    sn_list = [pickle.loads(sn) for sn in  list(pic_set)]
    sn_peers = sn_list
    return sn_peers
