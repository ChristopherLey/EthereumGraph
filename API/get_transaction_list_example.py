import os

from web3 import Web3

my_key = os.environ.get("ALCHEMY_API_KEY")
alchemy_url = f"https://eth-mainnet.g.alchemy.com/v2/{my_key}"
w3 = Web3(Web3.HTTPProvider(alchemy_url))

if w3.isConnected():
    ethereum = w3.eth
    latest_block = ethereum.block_number
    latest_block_details = ethereum.getBlock(latest_block)
    print("{")
    for key, item in latest_block_details.items():
        print("\t", key, ": ", item)
    print("}")
    for i, transaction_hash in enumerate(latest_block_details.transactions):
        transaction = ethereum.get_transaction(transaction_hash)
        print("{")
        for key, item in transaction.items():
            print(f"\t{key}: {item}")
        print("}")
else:
    print(f"Alchemy could not connect with {alchemy_url}")
