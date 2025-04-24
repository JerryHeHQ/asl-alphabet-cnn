import sys
sys.path.append('../')  # Adjust path as needed
from pycore.tikzeng import *

def to_CustomBlock(name, to, offset, caption, width=1, height=1, depth=1, color="gray"):
    return to_Conv(
        name=name, s_filer="", offset=offset, to=to,
        width=width, height=height, depth=depth,
        caption=caption, fill=color
    )

arch = [
    to_input('example.jpg'),

    # Conv Block 1
    to_ConvConvRelu(name='conv1', s_filer=100, n_filer=(32, 32), offset="(0,0,0)", to="(0,0,0)",
                    width=(2, 2), height=40, depth=40, caption="Conv1"),
    to_CustomBlock(name='bn1', to="(conv1-east)", offset="(0.2,0,0)", caption="BN", width=1, height=40, depth=40, color="blue!20"),
    to_Pool(name="pool1", offset="(0,0,0)", to="(bn1-east)", width=1, height=20, depth=20, opacity=0.5),

    # Conv Block 2
    to_ConvConvRelu(name='conv2', s_filer=50, n_filer=(64, 64), offset="(1.25,0,0)", to="(pool1-east)",
                    width=(2.5, 2.5), height=20, depth=20, caption="Conv2"),
    to_CustomBlock(name='bn2', to="(conv2-east)", offset="(0.2,0,0)", caption="BN", width=1, height=20, depth=20, color="blue!20"),
    to_Pool(name="pool2", offset="(0,0,0)", to="(bn2-east)", width=1, height=10, depth=10, opacity=0.5),

    # Conv Block 3
    to_ConvConvRelu(name='conv3', s_filer=25, n_filer=(128, 128), offset="(1.5,0,0)", to="(pool2-east)",
                    width=(3, 3), height=10, depth=10, caption="Conv3"),
    to_CustomBlock(name='bn3', to="(conv3-east)", offset="(0.2,0,0)", caption="BN", width=1, height=10, depth=10, color="blue!20"),
    to_Pool(name="pool3", offset="(0,0,0)", to="(bn3-east)", width=1, height=5, depth=5, opacity=0.5),

    # Conv Block 4
    to_ConvConvRelu(name='conv4', s_filer=12, n_filer=(256, 256), offset="(1.75,0,0)", to="(pool3-east)",
                    width=(4, 4), height=5, depth=5, caption="Conv4"),
    to_CustomBlock(name='bn4', to="(conv4-east)", offset="(0.2,0,0)", caption="BN", width=1, height=5, depth=5, color="blue!20"),
    to_Pool(name="pool4", offset="(0,0,0)", to="(bn4-east)", width=1, height=3, depth=3, opacity=0.5),

    # Fully Connected + Dropout
    to_FullyConnected(name="fc1", s_filer="", offset="(2,0,0)", to="(pool4-east)", width=1, height=1, depth=10, caption="FC1"),
    to_CustomBlock(name='dropout', to="(fc1-east)", offset="(0.2,0,0)", caption="Dropout", width=1, height=1, depth=10, color="red!20"),
    to_FullyConnected(name="fc2", s_filer="", offset="(1.5,0,0)", to="(dropout-east)", width=1, height=1, depth=5, caption="Output"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
