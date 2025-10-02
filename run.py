import sys
import server
import logging
import config

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, 'INFO'), datefmt='%H:%M:%S')

def main():
    config_path = r'C:\Users\15834\Documents\GitHub\FedCMOO\base_config.json'
    a = config.Config(config_path)
    s = server.Server(a)
    s.boot()
    s.train()
    print("No config json file provided!")

if __name__ == "__main__":
    main()