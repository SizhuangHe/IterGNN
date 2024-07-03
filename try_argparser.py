from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-a", "--a", nargs='+', type=int)
args = parser.parse_args()
print(args.a)
print(len(args.a))