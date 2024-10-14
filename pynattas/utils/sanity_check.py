import argparse, sys
from pynattas.builder.netBuilder import GenericNetwork


def parse_args():
    parser = argparse.ArgumentParser(description='Sanity check for GenericNetwork')
    parser.add_argument('--architecture', type=str, required=True, help='Network architecture')
    parser.add_argument('--input_channels', type=int, required=True, help='Number of input channels', default=3)
    parser.add_argument('--input_height', type=int, required=True, help='Input height', default=224)
    parser.add_argument('--input_width', type=int, required=True, help='Input width', default=224)
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes', default=2)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    
    architecture=args.architecture
    input_channels = args.input_channels
    input_height = args.input_height
    input_width = args.input_width
    num_classes = args.num_classes

    try:
        Net = GenericNetwork(
                architecture, 
                input_channels,
                input_height,
                input_width,
                num_classes,)

        Net.build()


        x = torch.randn(1, input_channels, input_height, input_width)
        Net(x)
        print(Net.get_param_size())
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)