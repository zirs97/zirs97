import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,default='./data/data_s12.npy')
    parser.add_argument('--base_save_path', type=str, default='./runs/')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument()