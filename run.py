import sys
import server
import logging
import config

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, 'INFO'), datefmt='%H:%M:%S')

def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        a = config.Config(config_path)
        s = server.Server(a)
        s.boot()
        t = s.train()
    else:
        print("No config json file provided!")

if __name__ == "__main__":
    main()