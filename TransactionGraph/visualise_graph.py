import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
from pyvis.network import Network
from pyetherscan.api import Connector
import yaml
import colorsys

config = yaml.load(open('../user_config.yaml', 'r'), yaml.Loader)
etherscan = Connector(config['etherscan_key'])

N = 10
HSV_tuples = [(float(x)/N, 1.0, 1.0) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

end_block_hex = etherscan.proxy.eth_blockNumber()['result']
end_block = int(end_block_hex, 16) - 20



nt = Network("800px", "800px")
node_hash = {}
number_nodes = 0

for i, rgb in enumerate(RGB_tuples):
    block_number = hex(end_block - i)
    block_details = etherscan.proxy.eth_getBlockByNumber(block_number)['result']
    for transaction in block_details['transactions']:
        if transaction['to'] is not None and transaction['from'] is not None:
            receiver = transaction['to']
            sender = transaction['from']
            if receiver not in node_hash:
                node_hash[receiver] = number_nodes
                nt.add_node(number_nodes, title=receiver, color='indigo')
                number_nodes += 1
            if sender not in nt.nodes:
                node_hash[sender] = number_nodes
                nt.add_node(number_nodes, title=sender, color='indigo')
                number_nodes += 1
            color = f'rgb({round(rgb[0]*255)}, {round(rgb[1]*255)}, {round(rgb[2]*255)})'
            nt.add_edge(node_hash[sender], node_hash[receiver], title=int(block_number, 16),
                        color={"color": color}, arrows='to')
print(f'Creating graph for blocks: {end_block - i}->{end_block}')
nt.show('transaction_graph.html')
